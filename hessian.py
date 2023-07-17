import numpy as np
from skimage.feature import hessian_matrix
from itertools import combinations_with_replacement

class Hessian:
    def __init__(self, input, scale) -> None:
        """
        Generates the Hessian matrix for a given image and scale.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.
        scale : int
            Expected structure width.
        sorted_by_abs : bool, optional
            If True, sorts Eigenvalues by their absolute values.
        white_ridges_rvr : bool, optional
            If True, swaps the sign of all Eigenvalues.


        Returns
        -------
        eigvals : ndarray
            Eigenvalues of the input.
        eigvects : ndarray, optional
            Eigenvectors of the input.
        """
        self.scale = scale
        self.eigenvalues, self.eigenvectors = self._compute_hessian(input)
        return

    def _compute_hessian(self, input):
        """
        Calculates eigenvalues and eigenvectors for the given input image.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.    

        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues of the input.
        eigenvectors : ndarray, optional
            Eigenvectors of the input.
        """
        hessian = hessian_matrix(input, sigma=self.scale)
        eigenvalues, eigenvectors = self._symmetric_compute_eigenvalues(hessian)

        return eigenvalues, eigenvectors
    
    def get_eigenvalues(self, sorted_by_abs=False, invert=False):
        """
        Returns the eigenvalues with optional preprocessing.

        Parameters
        ----------
        sorted_by_abs : bool, optional
            If True, sorts Eigenvalues by their absolute values.
        invert : bool, optional
            If True, swaps the sign of all Eigenvalues.

        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues of the input.
        """
        eigenvalues = self.eigenvalues

        if invert == True:
            self.eigenvalues = self.eigenvalues * -1
            eigenvalues = self.eigenvalues

        if sorted_by_abs == True:
            eigenvalues, _ = self._sort_by_abs()
            
        return eigenvalues

    def get_eigenvector_directions(self):
        """Caculates directions of the eigenvectors corresponding to the 
        eigenvalues with smallest absolute value."""
        _, eigenvectors = self._sort_by_abs()
        cartesians = eigenvectors[0,:,:,:]
        theta = np.arctan2(cartesians[:,:,1],cartesians[:,:,0])
        theta = np.rad2deg(theta)
        theta = np.where(theta < 0, theta + 180, theta) 
        return theta
    
    def _sort_by_abs(self):
        """
        Returns copies of eigenvalues and eigenvectors, sorted by their absolute value.
        """
        axis = 0
        # Create auxiliary array for indexing
        index = list(np.ix_(*[np.arange(i) for i in self.eigenvalues.shape]))

        # Get indices of abs sorted array
        index[axis] = np.abs(self.eigenvalues).argsort(axis)

        # Return abs sorted array
        return self.eigenvalues[tuple(index)], self.eigenvectors[tuple(index)]

    #Skimage functions, modified to also return eigenvectors
    def _symmetric_compute_eigenvalues(self, S_elems):
        """Custom Variant of Scikit-image code.

        References
        ----------
        https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/feature/corner.py#L383"""
        matrices = self._symmetric_image(S_elems)
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        eigs, vects = np.linalg.eigh(matrices)
        eigs = eigs[..., ::-1]
        vects = vects[..., ::-1, :]
        # transpose eigenvalues to shape (2, image.shape) and eigenvectors to (2, image.shape, 2)
        leading_axes_eig = tuple(range(eigs.ndim - 1))
        leading_axes_vects = tuple(range(vects.ndim - 1))
        return np.transpose(eigs, (eigs.ndim - 1,) + leading_axes_eig), np.transpose(vects, (vects.ndim - 1,) + leading_axes_vects)
        
    def _symmetric_image(self, S_elems):
        """Custom Variant of Scikit-image code

        References
        ----------
        https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/feature/corner.py#L417"""
        image = S_elems[0]
        symmetric_image = np.zeros(image.shape + (image.ndim, image.ndim),
                                dtype=S_elems[0].dtype)
        for idx, (row, col) in \
                enumerate(combinations_with_replacement(range(image.ndim), 2)):
            symmetric_image[..., row, col] = S_elems[idx]
            symmetric_image[..., col, row] = S_elems[idx]
        return symmetric_image

class Frangi:
    def __init__(self, scales, blobness, invert=False) -> None:
        """
        Filter by Frangi et al. [1]_ for 2D images.
        Parameters
        ----------
        scales : [int]
            List of expected structure widths.
        tau : float
            Parameter for calculation of normalized eigenvalues.
            High values will shrink filtered structures and low 
            values will expand filtered structures.
        invert : bool
            When the parameter is False, bright lines are filtered, otherwise 
            dark lines are filtered.

        References
        ----------
        .. [1] Frangi, A.F., Niessen, W.J., Vincken, K.L., Viergever, M.A. (1998). 
            Multiscale vessel enhancement filtering. In: Wells, W.M., Colchester, A., 
            Delp, S. (eds) Medical Image Computing and Computer-Assisted Intervention 
            — MICCAI’98. MICCAI 1998. Lecture Notes in Computer Science, vol 1496. 
            Springer, Berlin, Heidelberg.
            :DOI:`10.1007/BFb0056195`
        """
        self.scales = scales
        self.blobness = blobness
        self.invert = invert
        return
    
    def get_descriptor(self):
        return "frangi_scales{}-{}_blobness{}".format(np.min(self.scales), 
                                                      np.max(self.scales),
                                                      self.blobness)

    def compute(self, input, get_directions=False):
        """
        Computes filter results for the given image and optionally returns a 
        directionality map for the result.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.
        get_directions : bool, optional
            Returns the direction perpendicular to highest intensity curvature 
            for each pixel in addition to the filtered image.

        Returns
        -------
        out_image : 2D ndarray
            Filtered image.
        out_directions : 2D ndarray, optional
            Direction perpendicular to highest intensity curvature for each pixel.
        """
        results = np.empty(self.scales.shape + input.shape)
        out_image = np.empty(input.shape)
        directions = np.empty(self.scales.shape + input.shape)
        out_directions = np.empty(input.shape)

        #padding to catch and later cut off artifacts at image borders
        pad = int(np.max(self.scales)*2)
        pad_x = (pad, input.shape[0] + pad)
        pad_y = (pad, input.shape[1] + pad)
        input = np.pad(input, [pad_x, pad_y], mode='edge')

        for i, scale in enumerate(self.scales):
            hessian = Hessian(input, scale)
            eigenvals = hessian.get_eigenvalues(sorted_by_abs=True, invert=self.invert)

            norm = np.sqrt(eigenvals[0]**2 + eigenvals[1]**2)
            max_norm = np.max(norm)
        
            c = max_norm/2

            abs_eigen1 = np.abs(eigenvals[0])
            abs_eigen2 = np.abs(eigenvals[1]) + 1e-10 #avoid division by 0
            blob_ratios = abs_eigen1 / abs_eigen2
            result = np.where(eigenvals[1] > 0, 0, np.exp(-blob_ratios**2/2*self.blobness**2) * (1-np.exp(-norm**2/(2*c**2))))
            results[i] = result[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]
            if get_directions == True:
                directions[i] = hessian.get_eigenvector_directions()[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]
            
            print("Frangi: Finished scale = %i"%(scale))

        out_image = np.max(results, axis=0)

        if get_directions == True:
            max_indices = np.argmax(results, axis=0)
            for i in range(self.scales.size):
                out_directions = np.where(max_indices == i, directions[i,:,:], out_directions)             
            return out_image, out_directions
        else:
            return out_image


class Meijering:
    def __init__(self, scale, invert=False) -> None:
        """
        Filter by Meijering et al. [1]_ for 2D images.
        Parameters
        ----------
        scale : int
            Expected structure widths.
        invert : bool
            When the parameter is False, bright lines are filtered, otherwise 
            dark lines are filtered.

        References
        ----------
        .. [1] Frangi, A.F., Niessen, W.J., Vincken, K.L., Viergever, M.A. (1998). 
            Multiscale vessel enhancement filtering. In: Wells, W.M., Colchester, A., 
            Delp, S. (eds) Medical Image Computing and Computer-Assisted Intervention 
            — MICCAI’98. MICCAI 1998. Lecture Notes in Computer Science, vol 1496. 
            Springer, Berlin, Heidelberg.
            :DOI:`10.1007/BFb0056195`
        """
        self.scale = scale
        self.invert = invert
        return
    
    def get_descriptor(self):
        return "meijering_scale{}".format(self.scale)

    def compute(self, input, get_directions=False):
        """
        Computes filter results for the given image and optionally returns a 
        directionality map for the result.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.
        get_directions : bool, optional
            Returns the direction perpendicular to highest intensity curvature for 
            each pixel in addition to the filtered image.

        Returns
        -------
        out_image : 2D ndarray
            Filtered image.
        out_directions : 2D ndarray, optional
            Direction perpendicular to highest intensity curvature for each pixel.

        References
        ----------
        .. [1] Meijering, E., Jacob, M., Sarria, J.-.-C.F., Steiner, P., Hirling, H. and 
        Unser, M. (2004), Design and validation of a tool for neurite tracing and 
        analysis in fluorescence microscopy images. Cytometry, 58A: 167-176. 
        :DOI:`10.1002/cyto.a.20022`
        """
        out_image = np.empty(input.shape)

        #padding to catch and later cut off artifacts at image borders
        pad = int(self.scale*2)
        pad_x = (pad, input.shape[0] + pad)
        pad_y = (pad, input.shape[1] + pad)
        input = np.pad(input, [pad_x, pad_y], mode='edge')

        hessian = Hessian(input, self.scale)
        eigenvals = hessian.get_eigenvalues(invert=self.invert)

        mod_eigens_1 = eigenvals[0] - eigenvals[1]/3
        mod_eigens_2 = eigenvals[1] - eigenvals[0]/3
        max_mod_eigens = np.where(np.abs(mod_eigens_1) > np.abs(mod_eigens_2), mod_eigens_1, mod_eigens_2)
        min_mod_eigen = np.min(np.minimum(mod_eigens_1, mod_eigens_2))
        out_image = np.where(max_mod_eigens < 0, max_mod_eigens/min_mod_eigen, 0)
        out_image = out_image[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]

        if get_directions == True:
            out_directions = hessian.get_eigenvector_directions()[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]
            return out_image, out_directions
        else:
            return out_image

class Jerman:
    def __init__(self, scales, tau, invert) -> None:
        """
        Filter by Jerman et al. [1]_ for 2D images.
        Parameters
        ----------
        scales : [int]
            List of expected structure widths.
        tau : float
            Parameter for calculation of normalized eigenvalues.
            High values will shrink filtered structures and low 
            values will expand filtered structures.
        invert : bool
            When the parameter is False, bright lines are filtered, otherwise 
            dark lines are filtered.
        
        """
        self.scales = scales
        self.tau = tau
        self.invert = invert

    def get_descriptor(self):
        return "jerman_scales{}-{}_tau{}".format(np.min(self.scales), 
                                                      np.max(self.scales),
                                                      self.tau)

    def compute(self, input, get_directions=False):
        """
        Computes filter results for the given image and optionally returns a 
        directionality map for the result.

        Parameters
        ----------
        input : 2D ndarray
            Image that will be filtered.
        get_directions : bool, optional
            Returns the direction perpendicular to highest intensity curvature for 
            each pixel in addition to the filtered image.

        Returns
        -------
        out_image : 2D ndarray
            Filtered image.
        out_directions : 2D ndarray, optional
            Direction of lowest curvature for each pixel.
        """
        results = np.empty(self.scales.shape + input.shape)
        out_image = np.empty(input.shape)
        directions = np.empty(self.scales.shape + input.shape)
        out_directions = np.empty(input.shape)

        #padding to catch and later cut off artifacts at image borders
        pad = int(np.max(self.scales)*2)
        pad_x = (pad, input.shape[0] + pad)
        pad_y = (pad, input.shape[1] + pad)
        input = np.pad(input, [pad_x, pad_y], mode='edge')

        for i, scale in enumerate(self.scales):
            hessian = Hessian(input, scale)
            eigenvals = hessian.get_eigenvalues(sorted_by_abs=True, invert=not self.invert)

            eigen2 = eigenvals[1]

            threshold = self.tau * np.max(eigen2)
            
            regular_eigen = np.where(eigen2 <= threshold, threshold, 0)
            regular_eigen = np.where(0 < eigen2, 0, regular_eigen)
            regular_eigen = np.where(eigen2 > threshold, eigen2, regular_eigen)

            result =  np.where(eigen2 >= regular_eigen/2, 1, eigen2**2 * (regular_eigen - eigen2) * (3/(eigen2+regular_eigen))**2)
            result =  np.where(eigen2 <= 0, 0, result)
            result =  np.where(regular_eigen <= 0, 0, result)
            results[i] = result[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]
            if get_directions == True:
                directions[i] = hessian.get_eigenvector_directions()[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]

            print("Jerman: Finished scale %i"%scale)
        
        out_image = np.max(results, axis=0)

        if get_directions == True:
            max_indices = np.argmax(results, axis=0)
            for i in range(self.scales.size):
                out_directions = np.where(max_indices == i, directions[i,:,:], out_directions)             
            return out_image, out_directions
        else:
            return out_image