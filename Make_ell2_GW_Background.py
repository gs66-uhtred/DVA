import numpy as np
from . import Rotation_Functions as rf
import copy

class ell2_vector_harmonics(object):

    def __init__(self, theta_phis):
        #thetas and phis should be coordinates of star positions in a global frame.
        thetas = theta_phis[:,0]
        phis = theta_phis[:,1]
        self.thetas = thetas
        self.phis = phis
        #Spherical harmonics ordered from m=-2 to m=2.
        #Spherical basis is r-hat, theta-hat, phi-hat.
    
        self.Psi2s_spherical_vector = np.zeros((5, thetas.size, 3))
        pre_factor = (3./2.)*(5/np.pi)**0.5
        Psi2_theta_hat = np.zeros((5, thetas.size))
        Psi2_phi_hat = np.zeros((5, thetas.size))
        self.Psi2s_spherical_vector[2,:,1] = -pre_factor*np.sin(thetas)*np.cos(thetas)
        self.Psi2s_spherical_vector[3,:,1] = pre_factor*3.**-0.5*np.cos(2*thetas)*np.cos(phis)
        self.Psi2s_spherical_vector[1,:,1] = pre_factor*3.**-0.5*np.cos(2*thetas)*np.sin(phis)
        self.Psi2s_spherical_vector[3,:,2] = -pre_factor*3.**-0.5*np.cos(thetas)*np.sin(phis)
        self.Psi2s_spherical_vector[1,:,2] = pre_factor*3.**-0.5*np.cos(thetas)*np.cos(phis)
        self.Psi2s_spherical_vector[4,:,1] = pre_factor*3.**-0.5*np.cos(thetas)*np.sin(thetas)*np.cos(2*phis)
        self.Psi2s_spherical_vector[0,:,1] = pre_factor*3.**-0.5*np.cos(thetas)*np.sin(thetas)*np.sin(2*phis)
        self.Psi2s_spherical_vector[4,:,2] = -pre_factor*3.**-0.5*np.sin(thetas)*np.sin(2*phis)
        self.Psi2s_spherical_vector[0,:,2] = pre_factor*3.**-0.5*np.sin(thetas)*np.cos(2*phis)

        self.Phi2s_spherical_vector = np.zeros((5, thetas.size, 3))
        self.Phi2s_spherical_vector[2,:,2] = -pre_factor*np.sin(thetas)*np.cos(thetas)
        self.Phi2s_spherical_vector[3,:,2] = pre_factor*3.**-0.5*np.cos(2*thetas)*np.cos(phis)
        self.Phi2s_spherical_vector[1,:,2] = pre_factor*3.**-0.5*np.cos(2*thetas)*np.sin(phis)
        self.Phi2s_spherical_vector[3,:,1] = pre_factor*3.**-0.5*np.cos(thetas)*np.sin(phis)
        self.Phi2s_spherical_vector[1,:,1] = -pre_factor*3.**-0.5*np.cos(thetas)*np.cos(phis)
        self.Phi2s_spherical_vector[4,:,2] = pre_factor*3.**-0.5*np.cos(thetas)*np.sin(thetas)*np.cos(2*phis)
        self.Phi2s_spherical_vector[0,:,2] = pre_factor*3.**-0.5*np.cos(thetas)*np.sin(thetas)*np.sin(2*phis)
        self.Phi2s_spherical_vector[4,:,1] = pre_factor*3.**-0.5*np.sin(thetas)*np.sin(2*phis)
        self.Phi2s_spherical_vector[0,:,1] = -pre_factor*3.**-0.5*np.sin(thetas)*np.cos(2*phis)

    def generate_ell2_gw_signal(self, Psi2_magnitudes, Phi2_magnitudes, include_small_radial_component = True, project_to_xyz = True):
        self.gw_signal_ell2_spherical_vector = np.sum(Psi2_magnitudes[:,None,None]*self.Psi2s_spherical_vector, axis = 0)
        self.gw_signal_ell2_spherical_vector += np.sum(Phi2_magnitudes[:,None,None]*self.Phi2s_spherical_vector, axis = 0)
        if include_small_radial_component:
            shift_magnitude = np.sum(self.gw_signal_ell2_spherical_vector**2, axis = -1)**0.5
            sine_shift = np.sin(shift_magnitude)
            cosine_shift = np.cos(shift_magnitude)
            self.gw_signal_ell2_spherical_vector = sine_shift[:,None]*(self.gw_signal_ell2_spherical_vector/shift_magnitude[:,None])
            self.gw_signal_ell2_spherical_vector[:,0] = cosine_shift - 1.
        if project_to_xyz:
            spherical_vector = self.gw_signal_ell2_spherical_vector
            self.gw_signal_ell2_cartesian_vector =  rf.transform_spherical_vector_to_cartesian(spherical_vector, self.thetas, self.phis)


class gw_background(object):

    def __init__(self, gw_energy_per_log_f_at_25_hz, f_nyquist, delta_f, type = 'BHBB', H0 = 70):
        #gw_energy_per_log_f is dimensionless energy density in GW per logarithmic frequency unit at 1 Hz.
        #frequencies will be in Hz.
        self.H0 = H0/(3.086e13)
        self.gw_energy_per_log_f_at_25_hz = gw_energy_per_log_f_at_25_hz
        self.f_nyquist = f_nyquist
        self.delta_f = delta_f
        self.frequencies = np.arange(0, f_nyquist + delta_f, delta_f)
        non_zero_frequencies = copy.deepcopy(self.frequencies)
        non_zero_frequencies[0] = non_zero_frequencies[1]/10.
        if type == 'BHBB':
            self.index = 2./3.
        else:
            self.index = 1.
        self.gw_energy_per_log_f = np.zeros(self.frequencies.size)
        self.gw_energy_per_log_f[1:] = self.gw_energy_per_log_f_at_25_hz*(self.frequencies[1:]/25.)**self.index
        self.dn_theta_freq_variance = (1./(4*np.pi))*(self.H0**2)*(non_zero_frequencies**-3)*self.gw_energy_per_log_f*delta_f
        self.hc = ((3/(2*np.pi**2))*self.H0**2*non_zero_frequencies**-2*self.gw_energy_per_log_f)**0.5

        #Also computed the expected variance in time bins of size 1/2f_nyquist. 
        self.dt_1bin = 1./(2*f_nyquist)
        self.dn_theta_time_variance = np.sum(self.dn_theta_freq_variance[1:])

    def predict_full_sky_astrometric_gw_variance(self, astrometric_rms_per_root_second = ((1./3600.)*(np.pi/180.)*10**-3)*(3*60)**0.5, num_stars = 10**8, total_obs_time = None, d_f = None):
        if type(total_obs_time) == type(None):
            total_obs_time = 1./self.delta_f
        if type(d_f) == type(None):
            d_f = self.delta_f
        self.full_sky_astrometric_gw_variance_dn_freq = (astrometric_rms_per_root_second**4)/(num_stars**2*d_f*total_obs_time**3)
        self.full_sky_astrometric_gw_variance_hc = (self.full_sky_astrometric_gw_variance_dn_freq**0.5)*(self.hc*self.dn_theta_freq_variance**-0.5)

    def Compute_C2_frequency_values(self):
        #83 percent of the deflection power is due to the ell=2 mode.
        #For now, model entire signal power in ell=2.
        #My ell=2 spherical harmonics are normalized so that their angular integral is ell(ell+1) = 6 for ell=2. Variance is 6/4pi
        #Each ell will have 2*(2ell+1) E- and B-like vector spherical modes contributing power.
        self.C2_freq = 4*np.pi*self.dn_theta_freq_variance/(10*6)

    def realize_spherical_harmonic_amplitudes_freq_space(self):
        try:
            C2_freq = self.C2_freq
        except AttributeError:
            self.Compute_C2_frequency_values()
            C2_freq = self.C2_freq
        self.drawn_frequencies = np.fft.rfftfreq(2*self.frequencies.size - 1)*2*self.f_nyquist
        n = self.drawn_frequencies.size

        temp_ones = np.ones((10, self.drawn_frequencies.size))
        self.ell2_freq_sds = temp_ones*2**0.5*self.C2_freq[None,:]**0.5

        self.ell2_freq_amplitudes = np.zeros(self.ell2_freq_sds.shape, dtype = 'complex')
        self.ell2_freq_amplitudes[:, 1:] += np.random.normal(scale = self.ell2_freq_sds[:,1:])
        self.ell2_freq_amplitudes[:, 1:] += 1j*np.random.normal(scale = self.ell2_freq_sds[:,1:])
        self.ell2_freq_amplitudes[:, 0] += np.random.normal(scale = 2**0.5*self.ell2_freq_sds[:,0])

        self.ell2_time_amplitudes = n*np.fft.irfft(self.ell2_freq_amplitudes)
        self.ell2_times_seconds = (1./(2*self.f_nyquist))*np.arange(self.ell2_time_amplitudes.shape[-1])

        return self.ell2_freq_amplitudes
            
        
      