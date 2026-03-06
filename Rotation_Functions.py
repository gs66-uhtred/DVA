import numpy as np
from scipy.spatial.transform import Rotation

def theta_phis_to_xyzs(theta_phis):
    xyzs = np.zeros((theta_phis.shape[0], 3))
    xyzs[:,2] = np.cos(theta_phis[:,0])
    sin_thetas = np.sin(theta_phis[:,0])
    xyzs[:,0] = sin_thetas*np.cos(theta_phis[:,1])
    xyzs[:,1] = sin_thetas*np.sin(theta_phis[:,1])
    return xyzs

def theta_phi_to_xyz(theta_phi):
    xyz = np.zeros((3))
    xyz[2] = np.cos(theta_phi[0])
    sin_theta = np.sin(theta_phi[0])
    xyz[0] = sin_theta*np.cos(theta_phi[1])
    xyz[1] = sin_theta*np.sin(theta_phi[1])
    return xyz

def xyz_to_theta_phi(xyz, assume_normalized=True):
    if not assume_normalized:
        xyz *= np.sum(xyz**2)**-0.5
    theta_phi = np.zeros((2))
    theta_phi[0] = np.arccos(xyz[2])
    if xyz[1] != 0:
        theta_phi[1] = np.sign(xyz[1])*np.arccos(xyz[0]*np.sum(xyz[:2]**2)**-0.5)
    else:
        theta_phi[1] = 0
    return theta_phi

def xyzs_to_theta_phis(xyzs, assume_normalized=True):
    if not assume_normalized:
        xyzs = xyzs*(np.sum(xyzs**2, axis = -1)**-0.5)[:,None]
    theta_phis = np.zeros((xyzs.shape[0], 2))
    theta_phis[:,0] = np.arccos(xyzs[:,2])
    theta_phis[:,1] = np.sign(xyzs[:,1])*np.arccos(xyzs[:,0]*np.sum(xyzs[:,:2]**2, axis = -1)**-0.5)
    theta_phis[:,1][xyzs[:,1] == 0] = 0
    return theta_phis

def get_rotation_from_vector_direction_to_zhat(vector, above_90 = True):
    z_hat_direction = np.array([0,0,1])
    cross = np.cross(vector, z_hat_direction)
    cross_norm = np.linalg.norm(cross)
    angle = np.arcsin(cross_norm)
    if above_90:
        angle = np.pi - angle
    rotation_vector_to_z_hat = angle*cross/cross_norm
    Rotation_to_z_hat = Rotation.from_rotvec(rotation_vector_to_z_hat)
    return Rotation_to_z_hat

def get_rotation_from_guide_stars_to_zhat(guide_star_positions, above_90 = True):
    guide_star_mean = np.mean(guide_star_positions, axis = 0)
    guide_star_mean /= np.linalg.norm(guide_star_mean)
    return get_rotation_from_vector_direction_to_zhat(guide_star_mean)

def spin_measure(guide_star_template, guide_stars):
    #Assume rotation purely along z axis.
    template_xy_radii = np.sum(guide_star_template[:,:2]**2, axis = -1)**0.5
    diff = guide_stars - guide_star_template
    phi_hat = np.zeros(guide_star_template.shape)
    phi_hat[:,0] = -guide_star_template[:,1]
    phi_hat[:,1] = guide_star_template[:,0]
    phi_hat = phi_hat/template_xy_radii[:,None]
    d_phi = np.sum(phi_hat*diff, axis = -1)/template_xy_radii
    mean_d_phi = np.sum(d_phi*template_xy_radii**2)/np.sum(template_xy_radii**2)
    return np.arcsin(mean_d_phi)

def get_de_spin_rotator(guide_star_template, guide_stars):
    angle = spin_measure(guide_star_template, guide_stars)
    if angle != 0:
        de_spin_rotator = Rotation.from_rotvec(np.array([0,0,-1])*angle)
    else:
        de_spin_rotator = Rotation.from_matrix(np.eye(3))
    return de_spin_rotator

def transform_spherical_vector_to_cartesian(spherical_vector, theta, phi):
    #Assume spherical vector of shape (N_positions, 3).
    #Order of spherical vector spatial dimensions is r_hat, theta_hat, phi_hat.
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    matrix = np.array([[st*cp, ct*cp, -sp], [st*sp, ct*sp, cp], [ct, -st, np.zeros(cp.shape)]])
    return np.sum(matrix*spherical_vector.T[None,:,:], axis = 1).T