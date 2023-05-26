import numpy as np

class LHE:
    def __init__(self, area_num) -> None:
        """
        Adaptive Histogram Equalization filter as described in [1]_ for 16-bit 
        grayscale images.

        Parameters
        ----------
        area_num : int
            Number of sample points/areas for interpolation in x-/y-direction 
            respectively. Odd numbers will be converted to the next lower even 
            number.

        References
        ----------
        .. [1] Stephen M. Pizer, E. Philip Amburn, John D. Austin, Robert 
        Cromartie, Ari Geselowitz, Trey Greer, Bart ter Haar Romeny, John B. 
        Zimmerman, Karel Zuiderveld, Adaptive histogram equalization and its 
        variations, Computer Vision, Graphics, and Image Processing, Volume 39, 
        Issue 3, 1987, Pages 355-368, ISSN 0734-189X.
        :DOI:`10.1016/S0734-189X(87)80186-X`
        """
        if area_num%2 != 0:
            self.area_num = area_num - 1
        else:
            self.area_num = area_num

    def get_descriptor(self):
        return "lhe_areas{}".format(self.area_num)

    def compute(self, input, get_directions=False):
        """
        Computes filter results for the given image.
        
        Parameters
        ----------
        input : 2D ndarray
            Image that will be equalized.
        alpha : float, optional
            Specifies the influence of the local histogram and the histogram 
            of the remaining image on the result. Values below 0.5 will 
            emphasize the local histogram and values above 0.5 will emphasize 
            the histogram of the remaining image. 
            Takes values between 0 and 1.
        beta : float, optional
            Value at which the normalized Histogram is clipped. 
            Takes values between 0 and 1.
        get_directions : bool, optional
            Streamlines interface for use in pipelines.
        
        Returns
        -------
        output : 2D ndarray
            Equalized image.
        out_directions : None, optional
        """
        output = np.empty(input.shape)
        area_size = np.array([input.shape[0]//self.area_num + 1, input.shape[1]//self.area_num + 1])

        mapped_images = np.empty((2,2)+(input.shape))
        for i in range(self.area_num):
            k = i % 2
            for j in range(self.area_num):
                l = j % 2
                #define histogram regions and compute histograms
                area_start_x = i * area_size[0]
                area_start_y = j * area_size[1]
                area_end_x = np.clip(area_start_x+area_size[0], None, input.shape[0])
                area_end_y = np.clip(area_start_y+area_size[1], None, input.shape[1])
                area_mid_x = area_start_x + area_size[0]//2
                area_mid_y = area_start_y + area_size[1]//2
                histo_area = input[area_start_x:area_end_x,area_start_y:area_end_y]

                histo, bins = self._area_histogram(input, histo_area)            
                cdf = np.round((2**16-1) * np.cumsum(histo))         
                
                #define mapping regions and compute mapped images
                area_start_x = np.clip(area_mid_x - area_size[0], 0, None)
                area_start_y = np.clip(area_mid_y - area_size[1], 0, None)
                area_end_x = np.clip(area_mid_x + area_size[0], None, input.shape[0])
                area_end_y = np.clip(area_mid_y + area_size[1], None, input.shape[1])
                mapping_area = input[area_start_x:area_end_x,area_start_y:area_end_y]
                mapped_area = np.interp(mapping_area.flatten(), bins, cdf)
                mapped_area = mapped_area.reshape(mapping_area.shape)
                #m[0,0,area_start_x:area_end_x,area_start_y:area_end_y] = mapped_area
                mapped_images[k,l,area_start_x:area_end_x,area_start_y:area_end_y] = mapped_area

        triangle_x = np.array(list(self._triangle(area_size[0]*2, input.shape[0])))
        triangle_y = np.array(list(self._triangle(area_size[1]*2, input.shape[1])))
        weight_function = triangle_x[:, np.newaxis] *  triangle_y[:, np.newaxis].T
        
        #interpolate mapped images
        mapped_images = mapped_images.reshape(4,input.shape[0], input.shape[1])
        offset_directions = [[1,1],[1,0],[0,1],[0,0]]
        for i in range(4):
            mapped_image = mapped_images[i]
            weight_offsets = offset_directions[i] * (area_size - area_size%2)
            padded_weights = np.pad(weight_function, [(area_size[0]//2,area_size[0]//2),(area_size[1]//2,area_size[1]//2)])
            weights = padded_weights[weight_offsets[0]:input.shape[0]+weight_offsets[0], weight_offsets[1]:input.shape[1]+weight_offsets[1]]
            output = output + weights * mapped_image

        if get_directions == False:
            return output
        else:
            return output, None

    def _area_histogram(self, input, area):
        """
        Generates the histogram for the given area.

        Parameters
        ----------
        area : 2D ndarray
            Image area for which the histogram will be calculated.

        Returns
        -------
        histo : 1D ndarray
            Amount of existing values per bin.
        bins : 1D ndarray
            Unique values that exist in the given area.
        """
        histo, bins = np.histogram(area, bins=2**16, range=(0,2**16)) #normalize
        histo = histo / (area.shape[0] * area.shape[1])
        bins = bins[:2**16]
        return histo, bins

    def _triangle(self, period, list_length):
        """
        Generates a discrete triangle wave with values between 0 and 1.

        Parameters
        ----------
        period : int
            The number of values after which the triangle pattern is repeated.
        list_length : int
            Length of the returned wave snippet.

        Yields
        ------
        float
            Values of the triangle wave.
        """
        for i in range(list_length):
            j = i % period
            half = period/2
            if j < half:
                yield j / half
            if j >= half:
                yield 1 - (j - half) / half

class CLAHE(LHE):
    def __init__(self, area_num, beta) -> None:
        """
        Contrast Limited Adaptive Histogram Equalization [1]_ filter for 
        16-bit grayscale images.

        Parameters
        ----------
        area_num : int
            Number of sample points/areas for interpolation in x-/y-direction 
            respectively. Odd numbers will be converted to the next lower even 
            number.
        beta : float 
            Value at which the normalized Histogram is clipped. 
            Takes values between 0 and 1. 
        References
        ----------
        .. [1] Stephen M. Pizer, E. Philip Amburn, John D. Austin, Robert 
        Cromartie, Ari Geselowitz, Trey Greer, Bart ter Haar Romeny, John B. 
        Zimmerman, Karel Zuiderveld, Adaptive histogram equalization and its 
        variations, Computer Vision, Graphics, and Image Processing, Volume 39, 
        Issue 3, 1987, Pages 355-368, ISSN 0734-189X.
        :DOI:`10.1016/S0734-189X(87)80186-X`
        """
        super().__init__(area_num)
        self.beta = beta

    def get_descriptor(self):
        return "clahe_areas{}_beta{}".format(self.area_num, self.beta)
    
    def _area_histogram(self, input, area):
        """
        Generates the histogram with clipped contrast for the given area.

        Parameters
        ----------
        area : 2D ndarray
            Image area for which the histogram will be calculated.

        Returns
        -------
        histo : 1D ndarray
            Amount of existing values per bin.
        bins : 1D ndarray
            Unique values that exist in the given area.
        """
        histo, bins = np.histogram(area, bins=2**16, range=(0,2**16)) #normalize
        histo = histo / (area.shape[0] * area.shape[1])
        bins = bins[:2**16]
        #redistribute histogram counts that exceed beta
        excess = np.sum(np.where(histo > self.beta, histo - self.beta, 0))
        redist_amount = excess / np.count_nonzero(histo)
        histo = np.clip(histo, None, self.beta)
        for i in range(2**16):
            if histo[i] > 0:
                if histo[i] < self.beta - redist_amount:
                    excess = excess - redist_amount
                    histo[i] = histo[i] + redist_amount
                elif histo[i] < self.beta:
                    excess = excess - (self.beta - histo[i])
                    histo[i] = self.beta
        histo = np.where(histo != 0, histo + excess/np.count_nonzero(histo), histo)
        return histo, bins
    
class CLHE(LHE):
    def __init__(self, area_num, alpha) -> None:
        """
        Constrained Local Histogram Equalization [1]_ filter for 16-bit grayscale 
        images.

        Parameters
        ----------
        area_num : int
            Number of sample points/areas for interpolation in x-/y-direction 
            respectively. Odd numbers will be converted to the next lower even 
            number.
        alpha : float 
            Specifies the influence of the local histogram and the histogram 
            of the remaining image on the result. Values below 0.5 will 
            emphasize the local histogram and values above 0.5 will emphasize 
            the histogram of the remaining image. 
            Takes values between 0 and 1.
        References
        ----------
        .. [1] Hui Zhu, Francis H.Y. Chan, F.K. Lam, Image Contrast Enhancement by 
        Constrained Local Histogram Equalization, Computer Vision and Image 
        Understanding, Volume 73, Issue 2, 1999, Pages 281-290, ISSN 1077-3142.
        :DOI:`10.1006/cviu.1998.0723`
        """
        super().__init__(area_num)
        self.alpha = alpha

    def get_descriptor(self):
        return "clhe_areas{}_alpha{}".format(self.area_num, self.alpha)
    
    def _area_histogram(self, input, area):
        """
        Generates the weighted histogram for the given area.

        Parameters
        ----------
        area : 2D ndarray
            Image area for which the histogram will be calculated.

        Returns
        -------
        histo : 1D ndarray
            Amount of existing values per bin.
        bins : 1D ndarray
            Unique values that exist in the given area.
        """
        global_histo, _ = np.histogram(input, bins=2**16, range=(0,2**16))
        local_histo, bins = np.histogram(area, bins=2**16, range=(0,2**16))
        bins = bins[:2**16]
        rest_histo = global_histo - local_histo
        local_histo = local_histo / area.size #normalize
        rest_histo = rest_histo  / (input.size - area.size) #normalize
        histo = self.alpha * local_histo + (1 - self.alpha) * rest_histo 
        return histo, bins

class GHE:
    def __init__(self) -> None:
        """
        Global Histogram Equalization filter for 16-bit grayscale images.
        """
    
    def get_descriptor(self):
        return "ghe_areas".format(self.area_num)

    def compute(input):
        """
        Computes filter results for the given image.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be equalized.

        Returns
        -------
        output : 2D ndarray
            Equalized image.
        """
        histo, bins = np.histogram(input, bins=2**16, range=(0,2**16))
        histo = histo / (input.shape[0] * input.shape[1])
        bins = bins[:2**16]
        cdf = np.round((2**16-1) * np.cumsum(histo))
        output = np.interp(input.flatten(), bins, cdf)

        return output.reshape(input.shape)

