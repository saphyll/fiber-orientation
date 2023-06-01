import numpy as np
import matplotlib.pyplot as plt
from hessian import Hessian

class Log_Gabor:
    def __init__(self, orientations, scales) -> None:
        """
        Log Gabor filters for 2D images, as described in [1]_.

        Parameters
        ----------
        orientations : [int]
            Number of orientations for which phase congruency will be calculated.
        scales : [int]
            Scales of log-Gabor filters that will be included. Higher scales will
            lead to enhancement of larger structures.

        References
        ----------
        .. [1] Fischer, S., Šroubek, F., Perrinet, L. et al. Self-Invertible 2D 
        Log-Gabor Wavelets. Int J Comput Vis 75, 231–246 (2007). 
        https://doi.org/10.1007/s11263-006-0026-8
        """
        self.orientations = orientations
        self.scales = scales
    
    def get_descriptor(self):
        return "logGabor_scales{}-{}_orientations{}".format(np.min(self.scales), 
                                                                         np.max(self.scales),
                                                                         self.orientations.size)
 

    def compute(self, input):
        """
        Computes filter results for the given image.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.

        Returns
        -------
        filter_results : 4D ndarray
            Frequency domain filter responses, sorted by scale index and orientation 
            index.
        """
        input_fourier = np.fft.fft2(input)
        input_fourier = np.fft.fftshift(input_fourier)
        filter_shape = input_fourier.shape
        results_shape = self.scales.shape + self.orientations.shape + filter_shape
        filter_results = np.empty(results_shape, dtype=np.complex128)
        
        #matrix of coordiantes from center
        x, y = np.meshgrid(range(0, filter_shape[1]), range(0, filter_shape[0]))
        x = x  - np.floor(filter_shape[1]/2)
        y = y  - np.floor(filter_shape[0]/2)
        #set 0 to 1, as log(0) is not defined
        x[int(filter_shape[0]/2),int(filter_shape[1]/2)] = 1
        y[int(filter_shape[0]/2),int(filter_shape[1]/2)] = 1

        coord_angle, coord_scale = self._meshgrid_to_polar(x,y)

        for i, scale in enumerate(self.scales):
            for j, orientation in enumerate(self.orientations):
                center_scale = np.log2(filter_shape[0]) - scale
                center_angle = np.pi/self.orientations.shape[0] * orientation
                #center_angle = np.pi/orientations.shape[0] * (orientation + 0.5)
                #removed np.pi/orientations.shape[0] * (orientation + 0.5) for even scales
                #filters have to have the exact same orientation to work for phase congruency

                scale_bandwidth = 0.996 * np.sqrt(2/3)
                angle_bandwidth = 0.996 * 1/np.sqrt(2) * np.pi/self.orientations.shape[0]

                radial_component = np.exp(-1/2*((coord_scale-center_scale)/scale_bandwidth)**2)
                angular_component = np.exp(-1/2*((coord_angle-center_angle)/angle_bandwidth)**2)
                filter_part = radial_component * angular_component
                filter_results[i,j] = input_fourier*filter_part
            print("Finished scale %d: %f"%(i, scale))
        return filter_results

    def _meshgrid_to_polar(self, x, y):
        """
        Converts meshgrid coordinates to log-polar coordinates.
        """
        theta = np.arctan2(y,x)
        rho = np.log2(np.sqrt(x**2+y**2))
        return theta, rho


class PC:
    def __init__(self, orientations_num, scales, cutoff, gain, mode='sum') -> None:
        """
        Phase Congruency [1]_ for 2D images.

        Parameters
        ----------
        orientations_num : int
            Number of orientations for which phase congruency will be calculated.
        scales : [int]
            Scales of log-Gabor filters that will be included. Higher scales will
            lead to enhancement of larger structures.
        cutoff : float
            Value of frequency spread, below which the PC-value will be penalized.
            Accepts values between 0 and 1.
        gain : int
            Controls the sharpness of the cutoff through a sigmoid function.
            Accepts positive values.

        References
        ----------
        .. [1] Kovesi, Peter. “Image Features from Phase Congruency.” (1995).
        """
        self.orientations = np.arange(0, orientations_num)
        self.scales = scales
        self.cutoff = cutoff
        self.gain = gain
        self.mode = mode

    def get_descriptor(self):
        return "pc_scales{}-{}_orientations{}_cutoff{}_gain{}_mode{}".format(np.min(self.scales), 
                                                                         np.max(self.scales),
                                                                         self.orientations.size,
                                                                         self.cutoff,
                                                                         self.gain,
                                                                         self.mode)
        
    def compute(self, input, get_directions=False):
        """
        Computes filter results for the given image.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.
        mode : str, optional
            One of the following string values
            sum (default)
                Calculates the sum of phase congruencies over all orientations 
                and scales as defined in [1]_.
            max
                Calculates the maximum of phase congruency of all separate scales.
        get_directions : bool, optional
            Returns the direction perpendicular to highest intensity curvature 
            for each pixel in addition to the filtered image.
                
        Returns
        -------
        phase_congruencies : 2D ndarray
            Phase congruency values for the chosen mode.
        direction_map : 2D ndarray, optional
            Direction perpendicular to highest intensity curvature for each pixel. 
            This value is calculated using the hessian matrix of the image with the 
            maximum scale used for PC calculation.

        References
        ----------
        .. [1] Kovesi, Peter. “Image Features from Phase Congruency.” (1995).
        """
        if self.mode == 'sum':
            phase_congruencies = self._compute_pc(input)
            if get_directions == False:
                return phase_congruencies
            else:
                hessian_scale = np.max(self.scales) * np.max(self.scales)//2
                hessian = Hessian(input, hessian_scale)
                direction_map = hessian.get_eigenvector_directions()
                return phase_congruencies, direction_map

        if self.mode == 'max':
            phase_congruencies = np.empty(self.scales.shape + input.shape)
            for i, scale in enumerate(self.scales):
                phase_congruencies[i] = self._compute_pc(input, scale)
            phase_congruencies = np.max(phase_congruencies, axis=0)
            if get_directions == False:
                return phase_congruencies
            else:
                hessian_scale = np.max(self.scales) * np.max(self.scales)//2
                hessian = Hessian(input, hessian_scale)
                direction_map = hessian.get_eigenvector_directions()
                return phase_congruencies, direction_map

    def _compute_pc(self, input, scale=None):
        """
        Computes filter results for all scales or for a single scale.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.
        scale : int, optional
            Single scale for which phase congruency will be calculated.

        Returns
        -------
        phase_congruencies : 2D ndarray
            Phase congruency values for the given scale(s).
        """
        if scale == None:
            scales = self.scales
        else:
            scales = np.array([scale])

        log_gabor = Log_Gabor(self.orientations, scales)
        filter_results = log_gabor.compute(input)

        scales_num = scales.shape[0]
        processing_shape = scales.shape + input.shape
        results_shape = self.orientations.shape + input.shape
        scale_indices = range(scales_num)

        amplitudes = np.empty(processing_shape)
        phases = np.empty(processing_shape)
        phase_deviations = np.empty(processing_shape)
        weights = np.empty(processing_shape)
        weighted_deviations = np.empty(processing_shape)
        phase_congruencies = np.empty(results_shape)


        for o in self.orientations:
            print("Orientation: {}/{}".format(o+1,self.orientations.shape[0]))
            print("------------------")
            for s in scale_indices:
                filtered_image = filter_results[s,o]
                filtered_image = np.fft.ifftshift(filtered_image)
                filtered_even = np.fft.ifft2(filtered_image).real
                filtered_odd = np.fft.ifft2(filtered_image).imag
                amplitudes[s] = np.sqrt(filtered_even**2+filtered_odd**2)
                phases[s] = np.arctan2(filtered_odd, filtered_even)
            print("Finished amplitudes/phases.")

            mean_phases = np.mean(phases)
            noise_threshold = self._get_noise_threshold(amplitudes)
            print("Finished mean phases/noise threshold.")
            
            for s in scale_indices:
                phase_deviations[s] = np.cos(phases[s]-mean_phases) - np.abs(np.sin(phases[s]-mean_phases))
            weights = 1/(1+np.exp(self.gain*(self.cutoff-(1/scales_num)*(np.sum(amplitudes, axis=0)/(np.max(amplitudes) + 1e-10)))))
            print("Finished phase deviations/weights.")

            for s in scale_indices:
                weighted_deviations[s] = weights * np.clip(amplitudes[s]*phase_deviations[s]-noise_threshold, 0, None)
            

            phase_congruencies[o] = np.sum(weighted_deviations, axis=0) / (np.sum(amplitudes, axis=0) + 1e-10)
            print("Max congruency: {}".format(np.max(phase_congruencies[o])))
            print("Finished wheighted deviations/phase congruencies.")
            
        return np.sum(phase_congruencies, axis=0)
    
    def _get_noise_threshold(self, amplitudes):
        """
        Computes the noise threshold from the amplitude (energy) values of the 
        image, which are assumed to have a rayleigh distribution.
        """
        amplitude_sum = np.sum(amplitudes, axis=0)
        sigma_g = self._rayleighmode(amplitude_sum)
        mean_rayleigh = sigma_g * np.sqrt(np.pi/2)
        deviation_rayleigh = sigma_g * np.sqrt((4-np.pi)/2)
        return mean_rayleigh+2.5*deviation_rayleigh

    def _rayleighmode(self, data):
        """
        Mode of the histogram for the given data.
        (In this context it is assumed that the data has a rayleigh 
        distribution.)
        """
        histo, bins = np.histogram(data, data.size)
        max = bins[np.where(np.max(histo))]
        return max[0]