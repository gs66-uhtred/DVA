import numpy as np

#aberration_sh_rotator can compute the DVA signal up to 3rd order in Beta, for any peculiar velocity direction and magnitude and for any point on the sphere.
#The default arguments to the constructor will assume a peculiar velocity of 30 km/s and will compute all terms to 3rd order in Beta (which also means up to ell=3 vector spherical harmonics).
class aberration_sh_rotator(object):
    
    def __init__(self, beta = 10**-4, order = 3):
        if order>0:
            self.y10_initial = beta
        if order>1:
            self.y20_initial = -0.5*beta**2
        if order>2:
            self.y30_initial = (1./6.)*(2./5.)*beta**3
            self.y10_initial += (-1./6.)*(7./5.)*beta**3
    
    def generate_scalar_spherical_harmonics(self, thetas, phis):
        #ell = 1.
        self.Y10 = np.cos(thetas)
        self.Y11 = np.sin(thetas)*np.cos(phis)
        self.Y1n1 = np.sin(thetas)*np.sin(phis)
        #ell = 2.
        self.Y20 = 0.5*(3*np.cos(thetas)**2-1)
        self.Y21 = 0.5*3**0.5*np.sin(2*thetas)*np.cos(phis)
        self.Y2n1 = 0.5*3**0.5*np.sin(2*thetas)*np.sin(phis)
        self.Y22 = 0.5*3**0.5*np.sin(thetas)**2*np.cos(2*phis)
        self.Y2n2 = 0.5*3**0.5*np.sin(thetas)**2*np.sin(2*phis)
        #ell = 3.
        self.Y30 = 0.5*(5*np.cos(thetas)**3 - 3*np.cos(thetas))
        self.Y31 = 0.5*(3./2.)**0.5*(5*np.cos(thetas)**2*np.sin(thetas)*np.cos(phis) - np.sin(thetas)*np.cos(phis))
        self.Y3n1 = 0.5*(3./2.)**0.5*(5*np.cos(thetas)**2*np.sin(thetas)*np.sin(phis) - np.sin(thetas)*np.sin(phis))
        self.Y32 = 0.5*(15.)**0.5*(np.cos(phis)**2 - np.sin(phis)**2)*np.sin(thetas)**2*np.cos(thetas)
        self.Y3n2 = (15.)**0.5*(np.sin(phis)*np.cos(phis))*np.sin(thetas)**2*np.cos(thetas)
        self.Y33 = 0.5*(5./2.)**0.5*np.sin(thetas)**3*(np.cos(phis)**3 - 3*np.cos(phis)*np.sin(phis)**2)
        self.Y3n3 = 0.5*(5./2.)**0.5*np.sin(thetas)**3*(-np.sin(phis)**3 + 3*np.sin(phis)*np.cos(phis)**2)
        
    def generate_vector_spherical_harmonics(self, thetas, phis):
        #ell = 1.
        self.Psi10_theta_hat = -np.sin(thetas)
        self.Psi11_theta_hat = np.cos(thetas)*np.cos(phis)
        self.Psi1n1_theta_hat = np.cos(thetas)*np.sin(phis)
        self.Psi11_phi_hat = -np.sin(phis)
        self.Psi1n1_phi_hat = np.cos(phis)
        #ell = 2.
        self.Psi20_theta_hat = -np.sin(thetas)*np.cos(thetas)
        self.Psi21_theta_hat = 3.**-0.5*np.cos(2*thetas)*np.cos(phis)
        self.Psi2n1_theta_hat = 3.**-0.5*np.cos(2*thetas)*np.sin(phis)
        self.Psi21_phi_hat = -3.**-0.5*np.cos(thetas)*np.sin(phis)
        self.Psi2n1_phi_hat = 3.**-0.5*np.cos(thetas)*np.cos(phis)
        self.Psi22_theta_hat = 3.**-0.5*np.cos(thetas)*np.sin(thetas)*np.cos(2*phis)
        self.Psi2n2_theta_hat = 3.**-0.5*np.cos(thetas)*np.sin(thetas)*np.sin(2*phis)
        self.Psi22_phi_hat = -3.**-0.5*np.sin(thetas)*np.sin(2*phis)
        self.Psi2n2_phi_hat = 3.**-0.5*np.sin(thetas)*np.cos(2*phis)
        #ell = 3.
        self.Psi30_theta_hat = (np.sin(thetas) - 5*np.cos(thetas)**2*np.sin(thetas))
        self.Psi31_theta_hat = 3**-1*(3./2.)**0.5*np.cos(phis)*(5*np.cos(thetas)**3 - np.cos(thetas) - 10*np.cos(thetas)*np.sin(thetas)**2)
        self.Psi31_phi_hat = 3**-1*(3./2.)**0.5*(1 - 5*np.cos(thetas)**2)*np.sin(phis)
        self.Psi3n1_theta_hat = 3**-1*(3./2.)**0.5*np.sin(phis)*(5*np.cos(thetas)**3 - np.cos(thetas) - 10*np.cos(thetas)*np.sin(thetas)**2)
        self.Psi3n1_phi_hat = 3**-1*(3./2.)**0.5*(5*np.cos(thetas)**2 - 1)*np.cos(phis)
        self.Psi32_theta_hat = 3**-1*(15.)**0.5*(np.cos(phis)**2 - np.sin(phis)**2)*(2*np.sin(thetas)*np.cos(thetas)**2 - np.sin(thetas)**3)
        self.Psi32_phi_hat = -3**-1*(15.)**0.5*4*np.sin(phis)*np.cos(phis)*np.sin(thetas)*np.cos(thetas)
        self.Psi3n2_theta_hat = 3**-1*2*(15.)**0.5*np.sin(phis)*np.cos(phis)*(2*np.sin(thetas)*np.cos(thetas)**2 - np.sin(thetas)**3)
        self.Psi3n2_phi_hat = 3**-1*2*(15.)**0.5*(np.cos(phis)**2 - np.sin(phis)**2)*np.sin(thetas)*np.cos(thetas)
        self.Psi33_theta_hat = 3**-1*(5./2.)**0.5*3*np.sin(thetas)**2*np.cos(thetas)*(np.cos(phis)**3 - 3*np.cos(phis)*np.sin(phis)**2)
        self.Psi33_phi_hat = 3**-1*(5./2.)**0.5*(3*np.sin(phis)**3 - 9*np.cos(phis)**2*np.sin(phis))*np.sin(thetas)**2
        self.Psi3n3_theta_hat = 3**-1*(5./2.)**0.5*3*np.sin(thetas)**2*np.cos(thetas)*(-np.sin(phis)**3 + 3*np.sin(phis)*np.cos(phis)**2)
        self.Psi3n3_phi_hat = 3**-1*(5./2.)**0.5*(3*np.cos(phis)**3 - 9*np.sin(phis)**2*np.cos(phis))*np.sin(thetas)**2   
 
    def rotate_to_velocity_direction(self, theta_velocity, phi_velocity, thetas, phis):
        #Only generate vector spherical harmonic functions if they've not been generated already.
        try:
            Psi10_theta_hat = self.Psi10_theta_hat
        except AttributeError:
            self.generate_vector_spherical_harmonics(thetas, phis)
        #Generate scalar harmonic rotation coefficients for given velocity direction.
        self.generate_scalar_spherical_harmonics(theta_velocity, phi_velocity)
        #ell = 1.
        self.rotated_aberration_ell1_theta_hat = self.Y10*self.Psi10_theta_hat + self.Y11*self.Psi11_theta_hat
        self.rotated_aberration_ell1_theta_hat += self.Y1n1*self.Psi1n1_theta_hat
        self.rotated_aberration_ell1_phi_hat = self.Y11*self.Psi11_phi_hat
        self.rotated_aberration_ell1_phi_hat += self.Y1n1*self.Psi1n1_phi_hat
        self.rotated_aberration_ell1_phi_hat = self.rotated_aberration_ell1_phi_hat*self.y10_initial
        self.rotated_aberration_ell1_theta_hat = self.rotated_aberration_ell1_theta_hat*self.y10_initial
        self.rotated_aberration_ell1_magnitude = np.sqrt(self.rotated_aberration_ell1_theta_hat**2 + self.rotated_aberration_ell1_phi_hat**2)
        #ell = 2.
        self.rotated_aberration_ell2_theta_hat = self.Y20*self.Psi20_theta_hat + self.Y21*self.Psi21_theta_hat
        self.rotated_aberration_ell2_theta_hat += self.Y2n1*self.Psi2n1_theta_hat
        self.rotated_aberration_ell2_theta_hat += self.Y22*self.Psi22_theta_hat + self.Y2n2*self.Psi2n2_theta_hat
        self.rotated_aberration_ell2_phi_hat = self.Y21*self.Psi21_phi_hat
        self.rotated_aberration_ell2_phi_hat += self.Y2n1*self.Psi2n1_phi_hat
        self.rotated_aberration_ell2_phi_hat += self.Y22*self.Psi22_phi_hat + self.Y2n2*self.Psi2n2_phi_hat
        self.rotated_aberration_ell2_phi_hat *= self.y20_initial
        self.rotated_aberration_ell2_theta_hat *= self.y20_initial
        self.rotated_aberration_ell2_magnitude = np.sqrt(self.rotated_aberration_ell2_theta_hat**2 + self.rotated_aberration_ell2_phi_hat**2)
        #ell = 3.
        self.rotated_aberration_ell3_theta_hat = self.Y30*self.Psi30_theta_hat + self.Y31*self.Psi31_theta_hat
        self.rotated_aberration_ell3_theta_hat += self.Y3n1*self.Psi3n1_theta_hat
        self.rotated_aberration_ell3_theta_hat += self.Y32*self.Psi32_theta_hat + self.Y3n2*self.Psi3n2_theta_hat
        self.rotated_aberration_ell3_theta_hat += self.Y33*self.Psi33_theta_hat + self.Y3n3*self.Psi3n3_theta_hat
        self.rotated_aberration_ell3_phi_hat = self.Y31*self.Psi31_phi_hat
        self.rotated_aberration_ell3_phi_hat += self.Y3n1*self.Psi3n1_phi_hat
        self.rotated_aberration_ell3_phi_hat += self.Y32*self.Psi32_phi_hat + self.Y3n2*self.Psi3n2_phi_hat
        self.rotated_aberration_ell3_phi_hat += self.Y33*self.Psi33_phi_hat + self.Y3n3*self.Psi3n3_phi_hat
        self.rotated_aberration_ell3_phi_hat *= self.y30_initial
        self.rotated_aberration_ell3_theta_hat *= self.y30_initial
        self.rotated_aberration_ell3_magnitude = np.sqrt(self.rotated_aberration_ell3_theta_hat**2 + self.rotated_aberration_ell3_phi_hat**2)
        #total.
        self.full_rotated_aberration_theta_hat = self.rotated_aberration_ell1_theta_hat + self.rotated_aberration_ell2_theta_hat + self.rotated_aberration_ell3_theta_hat
        self.full_rotated_aberration_phi_hat = self.rotated_aberration_ell1_phi_hat + self.rotated_aberration_ell2_phi_hat + self.rotated_aberration_ell3_phi_hat
        self.full_rotated_aberration_magnitude = np.sqrt(self.full_rotated_aberration_theta_hat**2 + self.full_rotated_aberration_phi_hat**2)

#planar_orbit_velocity_direction is a class to compute the velocity direction as a function of time for an orbit in a plane.
#Currently only implemented for circular orbit.
#Default north pole of the orbital axis is the ecliptic north pole, as viewed in Galactic coordinates.
class planar_orbit_velocity_direction(object):
    
    def __init__(self, north_pole_theta = 29.81*np.pi/180., north_pole_phi = 93.38*np.pi/180):
        self.north_pole_theta = north_pole_theta
        self.north_pole_phi = north_pole_phi
        #Find a unit vector perpendicular to the input vector. First, cross y-hat with orbital north pole.
        #Note that this assumes the orbital north pole is not in the y-hat direction.
        self.u = np.array([-np.cos(north_pole_theta), 0, np.sin(north_pole_theta)*np.cos(north_pole_phi)])
        #u_norm = (np.cos(north_pole_theta)**2 + np.sin(north_pole_theta)**2*np.cos(north_pole_phi)**2)**0.5
        u_norm = np.sum(self.u**2)**0.5
        self.u = self.u/u_norm
        #Find the second orbital plane vector via uxnorth_pole.
        self.w = np.array([-np.sin(north_pole_theta)**2*np.sin(north_pole_phi)*np.cos(north_pole_phi),
                          np.cos(north_pole_theta)**2 + np.sin(north_pole_theta)**2*np.cos(north_pole_phi)**2,
                          -np.sin(north_pole_theta)*np.cos(north_pole_theta)*np.sin(north_pole_phi)])
        self.w = self.w/u_norm
        
    def compute_circular_orbit_vector_coeffs(self, time, orbital_period, initial_phase):
        #Circular orbit equation is u*cos(2*np.pi*(time/orbital_period + intial_phase)) +
        # w*sin(2*np.pi*(time/orbital_period + intial_phase))
        self.orbital_phases = 2*np.pi*time/orbital_period + initial_phase
        self.u_coeffs = np.cos(self.orbital_phases)
        self.w_coeffs = np.sin(self.orbital_phases)
        
    def compute_circular_orbit_vector(self, time, orbital_period, initial_phase):
        self.compute_circular_orbit_vector_coeffs(time, orbital_period, initial_phase)
        self.circular_orbit_vector = self.u*self.u_coeffs + self.w*self.w_coeffs
        
    def compute_circular_orbit_vectors(self, time, orbital_period, initial_phase):
        self.compute_circular_orbit_vector_coeffs(time, orbital_period, initial_phase)
        self.circular_orbit_vectors = self.u[:,None]*self.u_coeffs[None,:] + self.w[:,None]*self.w_coeffs[None,:]
        
    def compute_circular_orbit_theta_phi(self, time, orbital_period, initial_phase):
        self.compute_circular_orbit_vectors(time, orbital_period, initial_phase)
        self.circular_orbit_phis = np.arccos(self.circular_orbit_vectors[0]*(self.circular_orbit_vectors[0]**2 + self.circular_orbit_vectors[1]**2)**-0.5)
        sign_y = np.sign(self.circular_orbit_vectors[1])
        self.circular_orbit_phis[sign_y!=0] *= sign_y[sign_y!=0]
        self.circular_orbit_thetas = np.arccos(self.circular_orbit_vectors[2]*(self.circular_orbit_vectors[0]**2 + self.circular_orbit_vectors[1]**2 + self.circular_orbit_vectors[2]**2)**-0.5)
