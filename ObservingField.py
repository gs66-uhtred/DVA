import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
from . import Rotation_Functions as rf
import copy
from . import DVA_compute as DVA

class ObservingField(object):

    def __init__(self, center_theta_phi, fov_radians, num_guide_stars = 32):
        self.center_theta_phi = center_theta_phi
        self.center_xyz = rf.theta_phi_to_xyz(self.center_theta_phi)
        self.fov_radians = fov_radians
        #self.geom = np.sin(center_theta_phi[0])
        self.above_90 = center_theta_phi[0]>np.pi
        self.generate_guide_stars(num_stars = num_guide_stars)
        self.compute_ideal_field_rotation()

    def generate_guide_stars(self, num_stars = 32):
        star_angles_about_center = np.linspace(0, 2*np.pi, num_stars + 1)[:-1]
        self.original_guide_stars_theta_phi = np.zeros((num_stars,2))
        self.original_guide_stars_theta_phi += self.center_theta_phi
        self.ideal_centered_guide_stars_theta_phi = np.zeros((num_stars,2))
        for ind in range(num_stars):
            self.original_guide_stars_theta_phi[ind,0] += 0.9*0.5*self.fov_radians*np.sin(star_angles_about_center[ind])
            self.geom = np.sin(self.original_guide_stars_theta_phi[ind,0])
            self.original_guide_stars_theta_phi[ind,1] += 0.9*0.5*self.fov_radians*np.cos(star_angles_about_center[ind])/self.geom
            self.ideal_centered_guide_stars_theta_phi[ind,0] = 0.9*0.5*self.fov_radians
            self.ideal_centered_guide_stars_theta_phi[ind,1] = star_angles_about_center[ind]
        self.original_guide_stars_xyz = rf.theta_phis_to_xyzs(self.original_guide_stars_theta_phi)
        self.ideal_centered_guide_stars_xyz = rf.theta_phis_to_xyzs(self.ideal_centered_guide_stars_theta_phi)

    def generate_stars_uniform(self, stars_per_side = 32):
        self.num_stars = stars_per_side**2
        num_stars = self.num_stars
        x = np.linspace(-self.fov_radians/2., self.fov_radians/2., stars_per_side)
        y = np.linspace(-self.fov_radians/2., self.fov_radians/2., stars_per_side)
        mesh_x, mesh_y = np.meshgrid(x, y)
        self.stars_original_positions_theta_phi = np.zeros((num_stars, 2))
        self.stars_original_positions_theta_phi[:,0] = self.center_theta_phi[0] + mesh_x.flatten()
        self.geom = np.sin(self.stars_original_positions_theta_phi[:,0])
        self.stars_original_positions_theta_phi[:,1] = mesh_y.flatten()/self.geom
        self.stars_original_positions_theta_phi[:,1] += self.center_theta_phi[1]
        self.stars_original_positions_xyz = rf.theta_phis_to_xyzs(self.stars_original_positions_theta_phi)
        self.stars_current_positions_xyz = copy.deepcopy(self.stars_original_positions_xyz)
        self.stars_current_positions_theta_phi = copy.deepcopy(self.stars_original_positions_theta_phi)
        try:
            self.ideal_centered_original_stars_xyz = self.ideal_field_rotation.apply(self.stars_original_positions_xyz)
            self.ideal_centered_original_stars_theta_phi = rf.xyzs_to_theta_phis(self.stars_original_positions_xyz)
            self.generate_gradient_modes()
        except AttributeError:
            pass

    def generate_random_stars_uniform(self, num_stars = 1000):
        self.num_stars = num_stars
        self.stars_original_positions_theta_phi = np.zeros((num_stars, 2))
        self.stars_original_positions_theta_phi[:,0] = self.center_theta_phi[0] + np.random.uniform(low = -self.fov_radians/2., high = self.fov_radians/2., size = num_stars)
        self.geom = np.sin(self.stars_original_positions_theta_phi[:,0])
        self.stars_original_positions_theta_phi[:,1] = np.random.uniform(low = -self.fov_radians/2., high = self.fov_radians/2., size = num_stars)/self.geom
        self.stars_original_positions_theta_phi[:,1] += self.center_theta_phi[1]
        self.stars_original_positions_xyz = rf.theta_phis_to_xyzs(self.stars_original_positions_theta_phi)
        self.stars_current_positions_xyz = copy.deepcopy(self.stars_original_positions_xyz)
        self.stars_current_positions_theta_phi = copy.deepcopy(self.stars_original_positions_theta_phi)
        try:
            self.ideal_centered_original_stars_xyz = self.ideal_field_rotation.apply(self.stars_original_positions_xyz)
            self.ideal_centered_original_stars_theta_phi = rf.xyzs_to_theta_phis(self.stars_original_positions_xyz)
            self.generate_gradient_modes()
        except AttributeError:
            pass

    def compute_aberration_from_original_positions(self, boost_magnitude, boost_theta, boost_phi, boost_order = 3):
        self.DVA_guide_stars = DVA.aberration_sh_rotator(beta = boost_magnitude, order = boost_order)
        self.boost_direction_theta_phi = np.zeros((2))
        self.boost_direction_theta_phi[0] = boost_theta
        self.boost_direction_theta_phi[1] = boost_phi
        self.boost_direction_xyz = rf.theta_phi_to_xyz(self.boost_direction_theta_phi)
        self.DVA_guide_stars.rotate_to_velocity_direction(boost_theta, boost_phi, self.original_guide_stars_theta_phi[:,0], self.original_guide_stars_theta_phi[:,1])
        self.DVA_perturbation_guide_stars_xyz = rf.transform_spherical_vector_to_cartesian(self.DVA_guide_stars.aberration_spherical_vector,
                                   self.original_guide_stars_theta_phi[:,0],
                                                 self.original_guide_stars_theta_phi[:,1])

        self.DVA_field_stars = DVA.aberration_sh_rotator(beta = boost_magnitude, order = boost_order)
        self.DVA_field_stars.rotate_to_velocity_direction(boost_theta, boost_phi, self.stars_original_positions_theta_phi[:,0], self.stars_original_positions_theta_phi[:,1])
        self.DVA_perturbation_field_stars_xyz = rf.transform_spherical_vector_to_cartesian(self.DVA_field_stars.aberration_spherical_vector,
                                   self.stars_original_positions_theta_phi[:,0],
                                                 self.stars_original_positions_theta_phi[:,1])

    def generate_gradient_modes(self, custom_star_positions_xyz = None, enforce_orthonormality = True):
        if type(custom_star_positions_xyz) == type(None):
            centered_star_positions_xyz = self.ideal_centered_original_stars_xyz
        else:
            centered_star_positions_xyz = custom_star_positions_xyz
        self.xx_mode = np.zeros(centered_star_positions_xyz.shape)
        self.xx_mode[:,0] = centered_star_positions_xyz[:,0]
        self.xx_mode = self.xx_mode*(np.sum(self.xx_mode**2)**-0.5)
        self.xy_mode = np.zeros(centered_star_positions_xyz.shape)
        self.xy_mode[:,0] = centered_star_positions_xyz[:,1]
        if enforce_orthonormality:
            self.xy_mode -= np.sum(self.xy_mode*self.xx_mode)*self.xx_mode
        self.xy_mode = self.xy_mode*(np.sum(self.xy_mode**2)**-0.5)

        self.yy_mode = np.zeros(centered_star_positions_xyz.shape)
        self.yy_mode[:,1] = centered_star_positions_xyz[:,1]
        self.yy_mode = self.yy_mode*(np.sum(self.yy_mode**2)**-0.5)
        self.yx_mode = np.zeros(centered_star_positions_xyz.shape)
        self.yx_mode[:,1] = centered_star_positions_xyz[:,0]
        if enforce_orthonormality:
            self.yx_mode -= np.sum(self.yx_mode*self.yy_mode)*self.yy_mode
        self.yx_mode = self.yx_mode*(np.sum(self.yx_mode**2)**-0.5)
        self.all_gradient_modes = np.zeros((4, self.yx_mode.shape[0], 3))
        self.all_gradient_modes[0,:] = self.xx_mode
        self.all_gradient_modes[1,:] = self.xy_mode
        self.all_gradient_modes[2,:] = self.yy_mode
        self.all_gradient_modes[3,:] = self.yx_mode

    def project_onto_gradient_modes(self, with_DVA_fitting_also = True, also_remove_exact_version_of_linear_solution = True, boost_order = 3):
        self.diff_proj_to_grad = np.sum(np.sum(self.all_gradient_modes*self.measured_dx_dy_dz[None,:,:], axis = -1), axis = -1)
        if with_DVA_fitting_also:
            try:
                DVA_rotation_solver = self.DVA_rotation_solver
            except AttributeError:
                DVA_rotation_solver = linear_DVA_rotation_solver(self.ideal_centered_original_stars_xyz)
                DVA_rotation_solver.create_design_matrix()
                self.DVA_rotation_solver = DVA_rotation_solver
            DVA_rotation_solver.compute_star_movement(self.new_rotated_new_star_positions_xyz)
            DVA_rotation_solver.estimate_boost_and_rotation()
            self.relative_boost_linear_estimate_xyz = copy.deepcopy(DVA_rotation_solver.parameters[:3])
            self.relative_rotation_angle_linear_estimate = copy.deepcopy(DVA_rotation_solver.parameters[3:])
            self.relative_boost_linear_estimate_magnitude = np.sum(self.relative_boost_linear_estimate_xyz**2)**0.5
            self.relative_boost_direction_linear_estimate_theta_phi = rf.xyz_to_theta_phi(self.relative_boost_linear_estimate_xyz, assume_normalized=False)
            self.measured_dx_dy_dz_minus_DVA_rot_dx_dy_fit = copy.deepcopy(self.measured_dx_dy_dz)
            self.measured_dx_dy_dz_minus_DVA_rot_dx_dy_fit[:,0] -= DVA_rotation_solver.estimated_offset[:self.num_stars]
            self.measured_dx_dy_dz_minus_DVA_rot_dx_dy_fit[:,1] -= DVA_rotation_solver.estimated_offset[self.num_stars:]
            self.diff_minus_DVA_rot_fit_proj_to_grad = np.sum(np.sum(self.all_gradient_modes*self.measured_dx_dy_dz_minus_DVA_rot_dx_dy_fit[None,:,:], axis = -1), axis = -1)
            if also_remove_exact_version_of_linear_solution:
                self.relative_DVA_model = DVA.aberration_sh_rotator(beta = self.relative_boost_linear_estimate_magnitude, order = boost_order)
                boost_theta = self.relative_boost_direction_linear_estimate_theta_phi[0]
                boost_phi = self.relative_boost_direction_linear_estimate_theta_phi[1]
                self.relative_DVA_model.rotate_to_velocity_direction(boost_theta, boost_phi, self.ideal_centered_original_stars_theta_phi[:,0], self.ideal_centered_original_stars_theta_phi[:,1])
                #self.relative_DVA_model.rotate_to_velocity_direction(boost_theta, boost_phi, self.original_rotated_guide_star_positions_theta_phi[:,0], self.original_rotated_guide_star_positions_theta_phi[:,1])
                self.relative_DVA_perturbation_xyz = rf.transform_spherical_vector_to_cartesian(self.relative_DVA_model.aberration_spherical_vector,
                                   self.ideal_centered_original_stars_theta_phi[:,0],
                                                 self.ideal_centered_original_stars_theta_phi[:,1])
                self.relative_rotator = Rotation.from_rotvec(self.relative_rotation_angle_linear_estimate)
                self.exact_form_of_linear_solution_xyz = self.relative_rotator.apply(self.relative_DVA_perturbation_xyz + self.ideal_centered_original_stars_xyz)
                self.measured_xyz_minus_exact_form_of_linear_solution = self.new_rotated_new_star_positions_xyz - self.exact_form_of_linear_solution_xyz
                #self.measured_xyz_minus_exact_form_of_linear_solution_proj_to_grad = np.sum(np.sum(self.all_gradient_modes*self.measured_dx_dy_dz_minus_DVA_rot_dx_dy_fit[None,:,:], axis = -1), axis = -1)
                self.measured_xyz_minus_exact_form_of_linear_solution_proj_to_grad = np.sum(np.sum(self.all_gradient_modes*self.measured_xyz_minus_exact_form_of_linear_solution[None,:,:], axis = -1), axis = -1)
                
            
        

    def perturb_original_field_stars(self, d_xyz, apply_current_rotation_matrix = True, add_noise = False, noise_level = 1.e-3/(3600.)*np.pi/180.):
        self.stars_current_positions_xyz = self.stars_original_positions_xyz + d_xyz
        self.stars_current_positions_theta_phi = rf.xyzs_to_theta_phis(self.stars_current_positions_xyz)
        if add_noise:
            self.current_star_noise = realize_astrometric_noise_at_star_positions_xyz(self.stars_current_positions_xyz, self.stars_current_positions_theta_phi, noise_level)
            self.stars_current_positions_xyz = self.stars_original_positions_xyz + d_xyz + self.current_star_noise
            self.stars_current_positions_theta_phi = rf.xyzs_to_theta_phis(self.stars_current_positions_xyz)
        if apply_current_rotation_matrix:
            self.new_rotated_new_star_positions_xyz = self.full_new_rot.apply(self.stars_current_positions_xyz)
            self.new_rotated_new_star_positions_theta_phi = rf.xyzs_to_theta_phis(self.new_rotated_new_star_positions_xyz)
            #
            self.new_rotated_original_star_positions_xyz = self.full_new_rot.apply(self.stars_original_positions_xyz)
            self.new_rotated_original_star_positions_theta_phi = rf.xyzs_to_theta_phis(self.new_rotated_original_star_positions_xyz)
            self.new_rotated_new_star_positions_xyz = self.full_new_rot.apply(self.stars_current_positions_xyz)
            self.new_rotated_new_star_positions_theta_phi = rf.xyzs_to_theta_phis(self.new_rotated_new_star_positions_xyz)
            self.measured_dx_dy_dz = self.new_rotated_new_star_positions_xyz - self.ideal_centered_original_stars_xyz

    def perturb_original_guide_stars(self, d_xyz, get_rotation_matrix = True, rotate_to_new_guide_position = True, add_noise = False, noise_level = 1.e-3/(3600.)*np.pi/180.):
        self.update_guide_star_positions(self.original_guide_stars_xyz + d_xyz, get_rotation_matrix = get_rotation_matrix, rotate_to_new_guide_position = rotate_to_new_guide_position)
        if add_noise:
            self.current_guide_star_noise = realize_astrometric_noise_at_star_positions_xyz(self.new_guide_star_positions_xyz, self.new_guide_star_positions_theta_phi, noise_level)
            self.update_guide_star_positions(self.original_guide_stars_xyz + d_xyz + self.current_guide_star_noise, get_rotation_matrix = get_rotation_matrix, rotate_to_new_guide_position = rotate_to_new_guide_position)

    def compute_ideal_field_rotation(self):
        first_rot = rf.get_rotation_from_guide_stars_to_zhat(self.original_guide_stars_xyz, above_90 = self.above_90)
        intermediate_guide_stars_xyz = first_rot.apply(self.original_guide_stars_xyz)
        second_rot = rf.get_de_spin_rotator(self.ideal_centered_guide_stars_xyz, intermediate_guide_stars_xyz)
        self.ideal_field_rotation = second_rot*first_rot
        self.original_rotated_guide_star_positions_xyz = self.ideal_field_rotation.apply(self.original_guide_stars_xyz)
        self.original_rotated_guide_star_positions_theta_phi = rf.xyzs_to_theta_phis(self.original_rotated_guide_star_positions_xyz)

    def update_guide_star_positions(self, new_guide_star_positions, get_rotation_matrix = True, rotate_to_new_guide_position = True):
        if new_guide_star_positions.shape[-1] == 2:
            #Guide stars are in theta, phi coordinates.
            self.new_guide_star_positions_theta_phi = new_guide_star_positions
            self.new_guide_star_positions_xyz = rf.theta_phis_to_xyzs(self.new_guide_star_positions_theta_phi)
        else:
            #Guide stars are in xyz coordinates.
            self.new_guide_star_positions_xyz = new_guide_star_positions
            self.new_guide_star_positions_theta_phi = rf.xyzs_to_theta_phis(self.new_guide_star_positions_xyz)
        if get_rotation_matrix:
            self.new_rot1 = rf.get_rotation_from_guide_stars_to_zhat(self.new_guide_star_positions_xyz, above_90 = self.above_90)
            self.intermediate_rotated_new_guide_star_positions_xyz = self.new_rot1.apply(self.new_guide_star_positions_xyz)
            self.new_rot2 = rf.get_de_spin_rotator(self.original_rotated_guide_star_positions_xyz, self.intermediate_rotated_new_guide_star_positions_xyz)
            self.full_new_rot = self.new_rot2*self.new_rot1
            if rotate_to_new_guide_position:
                self.rotated_new_guide_star_positions_xyz = self.full_new_rot.apply(self.new_guide_star_positions_xyz)
                self.rotated_new_guide_star_positions_theta_phi = rf.xyzs_to_theta_phis(self.rotated_new_guide_star_positions_xyz)
                try:
                    self.new_rotated_original_star_positions_xyz = self.full_new_rot.apply(self.stars_original_positions_xyz)
                    self.new_rotated_original_star_positions_theta_phi = rf.xyzs_to_theta_phis(self.new_rotated_original_star_positions_xyz)
                    self.new_rotated_new_star_positions_xyz = self.full_new_rot.apply(self.stars_current_positions_xyz)
                    self.new_rotated_new_star_positions_theta_phi = rf.xyzs_to_theta_phis(self.new_rotated_new_star_positions_xyz)
                    self.measured_dx_dy_dz = self.new_rotated_new_star_positions_xyz - self.ideal_centered_original_stars_xyz
                except:
                    pass

class linear_DVA_rotation_solver(object):

    def __init__(self, initial_centered_star_positions_xyz):
        self.initial_centered_star_positions_xyz = initial_centered_star_positions_xyz
        self.num_stars = initial_centered_star_positions_xyz.shape[0]

    def compute_star_movement(self, new_centered_star_positions_xyz):
        self.new_centered_star_positions_xyz = new_centered_star_positions_xyz
        self.dstar_xyz = self.new_centered_star_positions_xyz - self.initial_centered_star_positions_xyz
        self.dx_and_dy = np.zeros((2*self.num_stars))
        self.dx_and_dy[:self.num_stars] = self.dstar_xyz[:,0]
        self.dx_and_dy[self.num_stars:] = self.dstar_xyz[:,1]

    def create_design_matrix(self):
        num_stars = self.num_stars
        xi = self.initial_centered_star_positions_xyz[:,0]
        yi = self.initial_centered_star_positions_xyz[:,1]
        zi = self.initial_centered_star_positions_xyz[:,2]
        design_matrix = np.zeros((2*num_stars, 6))
        design_matrix[:num_stars,0] = (1 - xi**2)
        design_matrix[:num_stars,1] = -xi*yi
        design_matrix[:num_stars,2] = -xi*zi
        design_matrix[:num_stars,3] = -yi
        design_matrix[:num_stars,4] = zi
    
        design_matrix[num_stars:,0] = -xi*yi
        design_matrix[num_stars:,1] = (1. - yi**2)
        design_matrix[num_stars:,2] = -yi*zi
        design_matrix[num_stars:,3] = xi
        design_matrix[num_stars:,5] = -zi

        #design_matrix[:num_stars,0][zi==0] = 0
        #design_matrix[num_stars:,1][zi==0] = 0
        self.design_matrix = design_matrix

    def param_cov(self, star_error = (1.e-3)*np.pi/(3600.*180.), rotate_axes = False):
        design_mat = self.design_matrix
        cov = star_error**2*sp.linalg.inv(np.matmul(design_mat.T, design_mat))
        return cov

    def estimate_boost_and_rotation(self, weights = None):
        design_mat = self.design_matrix
        data = self.dx_and_dy
        if type(weights) == type(None):
            inverse = sp.linalg.inv(np.matmul(design_mat.T, design_mat))
            parameters = np.matmul(inverse, (np.matmul(design_mat.T, data)))
        else:
            inverse = sp.linalg.inv(np.matmul(np.matmul(design_mat.T, weights), design_mat))
            parameters = np.matmul(inverse, (np.matmul(design_mat.T, np.matmul(weights, data))))
        estimated_offset = np.matmul(design_mat, parameters)
        self.parameters = parameters
        self.estimated_offset = estimated_offset

def realize_astrometric_noise_at_star_positions_xyz(star_positions_xyz, star_positions_theta_phi, noise_level_radians):
    sigma = noise_level_radians*2**0.5
    num_stars = star_positions_xyz.shape[0]
    theta_shifts = np.random.normal(scale = sigma, size = num_stars)
    phi_shifts = np.random.normal(scale = sigma, size = num_stars)
    r_shifts = (1 - theta_shifts**2 - phi_shifts**2)**0.5 - 1
    spherical_vector_noise_shifts = np.zeros(star_positions_xyz.shape)
    spherical_vector_noise_shifts[:,0] = r_shifts
    spherical_vector_noise_shifts[:,1] = theta_shifts
    spherical_vector_noise_shifts[:,2] = phi_shifts
    xyz_noise_shifts = rf.transform_spherical_vector_to_cartesian(spherical_vector_noise_shifts,
                                   star_positions_theta_phi[:,0],
                                                 star_positions_theta_phi[:,1])
    return xyz_noise_shifts

