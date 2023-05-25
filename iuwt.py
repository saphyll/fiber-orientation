import numpy as np
import cv2

class IUWT:
    def __init__(self, scales, threshold) -> None:
        """
        Isotropic Undecimated Wavelet Transform as described by Meijering et al. [1]_ 
        for 2D images.
        Parameters
        ----------
        scales : 1D ndarray, dtype=int
            Filter sizes that are included in the output.
        threshold : int
            Percentage of (low) values that will be excluded from the output.

        References
        ----------
        .. [1] Bankhead P, Scholfield CN, McGeown JG, Curtis TM (2012) Fast Retinal 
        Vessel Detection and Measurement Using Wavelets and Edge Location Refinement. 
        PLoS ONE 7(3): e32435.
            :DOI:`10.1371/journal.pone.0032435`
        """
        self.scales = scales
        self.threshold = threshold
    
    def get_descriptor(self):
        return "iuwt_scales{}-{}_threshold{}".format(np.min(self.scales), 
                                                     np.max(self.scales),
                                                     self.threshold)
        
    def compute(self, input, get_directions=False):
        """
        Computes filter results for the given image and optionally returns a 
        directionality map for the result.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.
        get_directions : bool, optional
            Streamlines interface for use in pipelines.

        Returns
        -------
        out_image : 2D ndarray
            Filtered image.
        out_directions : None, optional
        """
        output = np.zeros(input.shape)
        kernels = self._get_scaled_filters()
        scale_coefficients = np.empty((np.max(self.scales),) + input.shape)
        wavelet_coefficients = []
        
        for scale in range(np.max(self.scales)):
            if scale == 0:
                scale_coefficients[scale] = cv2.filter2D(input, -1, kernels[scale])
            else:
                scale_coefficients[scale] = cv2.filter2D(scale_coefficients[scale-1], -1, kernels[scale])
                wavelet_coefficient = scale_coefficients[scale-1] - scale_coefficients[scale]
                wavelet_coefficients.append(wavelet_coefficient)
                if scale+1 in self.scales:
                    output = output + wavelet_coefficient

        percentile = np.percentile(output, self.threshold)
        output = np.where(output > percentile, output, np.min(output))

        if get_directions == False:
            return output
        else:
            return output, None

    def _get_scaled_filters(self):
        """
        Creates (upsampled) filter kernels, which are derived from the cubic B-spline.
        """
        base_filter = np.array([[1,4,6,4,1]])/16
        filters = [base_filter]
        kernels = [base_filter*base_filter.T]
        for scale in range(1, np.max(self.scales)+1):
            scaled_filter = np.zeros((1,(2**scale)*base_filter.shape[1] - (2**scale-1)))
            scaled_filter[:,::2**scale] = base_filter
            filters.append(scaled_filter)
            kernels.append(scaled_filter*scaled_filter.T)
        return kernels