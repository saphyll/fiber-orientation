import numpy as np
import cv2
from enum import Enum

class kernel_type(Enum):
    GK = 0
    AGK = 1
    FOAGK = 2
    SOAGK = 3

class AGK:
    def __init__(self, scales, anisotropies, orientations_num, invert=False) -> None:
        """
        Zero order (anisotropic) Gaussian Filter by Lopez-Molina et al. [1]_ for 2D images.
        Parameters
        ----------
        scales : [int]
            Expected structure widths.
        anisotropies : [float]
            Anisotropy values.
        orientations_num : int
            Number of included directions.

        Returns
        -------
        kernels : [[2D ndarray]]
            List of Kernels, ordered by orientation.

        References
        ----------
        .. [1] C. Lopez-Molina, G. Vidal-Diez de Ulzurrun, J.M. Baetens, J. Van den Bulcke, B. De Baets,
            Unsupervised ridge detection using second order anisotropic Gaussian kernels,
            Signal Processing, Volume 116, 2015, Pages 55-67, ISSN 0165-1684.
            :DOI:`10.1016/j.sigpro.2015.03.024`
        """
        self.scales = scales 
        self.anisotropies = anisotropies
        self.orientations = np.arange(0, (orientations_num)*np.pi/orientations_num, np.pi/orientations_num)
        self.kernels = self._create_kernels()
        self.invert = invert

    def _create_kernels(self):
        kernels = []
        for orientation in self.orientations:
            kernels_per_orientation = []
            for scale in self.scales:
                kernel_size = max(self.scales)*6 + 1
                kernel = np.zeros((kernel_size, kernel_size))
                for anisotropy in self.anisotropies:
                    it = np.nditer(kernel, flags=['multi_index'])
                    for _ in it:
                        kernel_index_x = it.multi_index[0] - np.floor(kernel_size/2)
                        kernel_index_y = it.multi_index[1] - np.floor(kernel_size/2)
                        x, y = self._phi(orientation, anisotropy, kernel_index_x, kernel_index_y)
                        kernel[it.multi_index[0], it.multi_index[1]] = self._gk(scale, x, y)
                    kernels_per_orientation.append(self._normalize(kernel))
            kernels.append(kernels_per_orientation)
        return kernels
    
    def get_descriptor(self):
        return "agk_scales{}-{}_anisotropies{}-{}_orientations{}".format(np.min(self.scales), 
                                                                         np.max(self.scales),
                                                                         np.min(self.anisotropies),
                                                                         np.max(self.anisotropies),
                                                                         self.orientations.size)

    def compute(self, input, get_directions=False):
        """
        Computes filter results for the given image and optionally returns a 
        directionality map for the result.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.
        invert : bool
            Inverts kernel intensities for detection of bright lines.
        get_directions : bool, optional
            Returns the direction perpendicular to highest curvature for each 
            pixel in addition to the filtered image.

        Returns
        -------
        out_image : 2D ndarray
            Filtered image.
        out_directions : 2D ndarray, optional
            Direction of lowest curvature for each pixel.
        """
        output = np.empty(input.shape)
        angleMap = np.empty(input.shape)
        for o, orientation in enumerate(self.orientations):
            for kernel in self.kernels[o]:
                    #invert kernel for detection of bright lines
                    if self.invert == False:
                        kernel = kernel * -1

                    kernel_output = cv2.filter2D(input, -1, kernel)
                    
                    for i in range(output.shape[0]):
                        indices = np.where(kernel_output[i] > output[i])
                        for j in indices[0]:
                            output[i][j] = kernel_output[i][j]
                            angleMap[i][j] = orientation 
        #normalize output
        output = output - np.min(output)
        output = output / np.max(output) 

        if get_directions == True:
            angleMap = np.rad2deg(angleMap) + .5 # +.5 to avoid wrong bin assignments in histograms
            return output, angleMap
        else:
            return output
    
    def _gk(self, sigma, x, y):
        """
        Computes values of a zero order gaussian kernel for the given coordinates.
        """
        return 1/2 * np.pi * sigma**2 * np.exp(-(x**2+y**2)/(2*sigma**2))

    def _fogk(self, sigma, x, y):
        """
        Computes values of a first order gaussian kernel for the given coordinates.
        """
        return -x/sigma**2 * self._gk(sigma, x, y)

    def _sogk(self, sigma, x, y):
        """
        Computes values of a second order gaussian kernel for the given coordinates.
        """
        gk_val = self._gk(sigma, x, y)
        return (-x/sigma**2) * (-x/sigma**2) * gk_val - (gk_val/sigma**2)
    
    def _phi(self, theta, rho, x, y):
        """
        Rotates coordinates by a given angle theta and spreads them by anisotropy value rho.
        """
        values = np.linalg.multi_dot([[x,y], self._rotation(theta), [[rho**2, 0], [0, rho**-2]]])
        return values[0], values[1]

    def _rotation(self, theta):
        """
        Rotation matrix for a given angle theta.
        """
        return [[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]
    
    def _normalize(self, kernel):
        return kernel / np.sum(np.abs(kernel))

class FOAGK(AGK):
    def __init__(self, scales, anisotropies, orientations_num, invert=False) -> None:
        """
        First order (anisotropic) Gaussian Filter by Lopez-Molina et al. [1]_ for 2D images.
        Parameters
        ----------
        scales : [int]
            Expected structure widths.
        anisotropies : [float]
            Anisotropy values.
        orientations_num : int
            Number of included directions.

        Returns
        -------
        kernels : [[2D ndarray]]
            List of Kernels, ordered by orientation.

        References
        ----------
        .. [1] C. Lopez-Molina, G. Vidal-Diez de Ulzurrun, J.M. Baetens, J. Van den Bulcke, B. De Baets,
            Unsupervised ridge detection using second order anisotropic Gaussian kernels,
            Signal Processing, Volume 116, 2015, Pages 55-67, ISSN 0165-1684.
            :DOI:`10.1016/j.sigpro.2015.03.024`
        """
        super().__init__(scales, anisotropies, orientations_num, invert)

    def get_descriptor(self):
        return "foagk_scales{}-{}_anisotropies{}-{}_orientations{}".format(np.min(self.scales), 
                                                                         np.max(self.scales),
                                                                         np.min(self.anisotropies),
                                                                         np.max(self.anisotropies),
                                                                         self.orientations.size)
    
    def _create_kernels(self):
        kernels = []
        for orientation in self.orientations:
            kernels_per_orientation = []
            for scale in self.scales:
                kernel_size = max(self.scales)*6 + 1
                kernel = np.zeros((kernel_size, kernel_size))
                for anisotropy in self.anisotropies:
                    it = np.nditer(kernel, flags=['multi_index'])
                    for _ in it:
                        kernel_index_x = it.multi_index[0] - np.floor(kernel_size/2)
                        kernel_index_y = it.multi_index[1] - np.floor(kernel_size/2)
                        x, y = self._phi(orientation, anisotropy, kernel_index_x, kernel_index_y)
                        kernel[it.multi_index[0], it.multi_index[1]] = self._fogk(scale, x, y)
                    kernels_per_orientation.append(self._normalize(kernel))
            kernels.append(kernels_per_orientation)
        return kernels

class SOAGK(AGK):
    def __init__(self, scales, anisotropies, orientations_num, invert=False) -> None:
        """
        Second order (anisotropic) Gaussian Filter by Lopez-Molina et al. [1]_ for 2D images.
        Parameters
        ----------
        scales : [int]
            Expected structure widths.
        anisotropies : [float]
            Anisotropy values.
        orientations_num : int
            Number of included directions.
        
        Returns
        -------
        kernels : [[2D ndarray]]
            List of Kernels, ordered by orientation.

        References
        ----------
        .. [1] C. Lopez-Molina, G. Vidal-Diez de Ulzurrun, J.M. Baetens, J. Van den Bulcke, B. De Baets,
            Unsupervised ridge detection using second order anisotropic Gaussian kernels,
            Signal Processing, Volume 116, 2015, Pages 55-67, ISSN 0165-1684.
            :DOI:`10.1016/j.sigpro.2015.03.024`
        """
        super().__init__(scales, anisotropies, orientations_num, invert)

    def get_descriptor(self):
        return "soagk_scales{}-{}_anisotropies{}-{}_orientations{}".format(np.min(self.scales), 
                                                                         np.max(self.scales),
                                                                         np.min(self.anisotropies),
                                                                         np.max(self.anisotropies),
                                                                         self.orientations.size)

    def _create_kernels(self):
        kernels = []
        for orientation in self.orientations:
            kernels_per_orientation = []
            for scale in self.scales:
                kernel_size = max(self.scales)*6 + 1
                kernel = np.zeros((kernel_size, kernel_size))
                for anisotropy in self.anisotropies:
                    it = np.nditer(kernel, flags=['multi_index'])
                    for _ in it:
                        kernel_index_x = it.multi_index[0] - np.floor(kernel_size/2)
                        kernel_index_y = it.multi_index[1] - np.floor(kernel_size/2)
                        x, y = self._phi(orientation, anisotropy, kernel_index_x, kernel_index_y)
                        kernel[it.multi_index[0], it.multi_index[1]] = self._sogk(scale, x, y)
                    kernels_per_orientation.append(self._normalize(kernel))
            kernels.append(kernels_per_orientation)
        return kernels