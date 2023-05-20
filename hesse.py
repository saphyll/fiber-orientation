import numpy as np
from skimage.feature import hessian_matrix
from itertools import combinations_with_replacement

def compute_frangi(input, scales, blobness=0.5, get_directions=False):
    """
    Frangi Filter [1]_ for 2D images.

    Parameters
    ----------
    input : 2D ndarray
        Image that will be filtered.
    scales : [int]
        List of expected structure widths.
    blobness : float
        Ratio at which blob-like structures will be included in the result.
        Low values will cause blob-like structures to fade.
        Accepts values between 0 and 1.
    get_directions : bool, optional
        Returns the direction of lowest curvature for each pixel in addition 
        to the filtered image.


    Returns
    -------
    out_image : 2D ndarray
        Filtered image.
    out_directions : 2D ndarray, optional
        Direction of lowest curvature for each pixel.

    References
    ----------
    .. [1] Frangi, A.F., Niessen, W.J., Vincken, K.L., Viergever, M.A. (1998). 
        Multiscale vessel enhancement filtering. In: Wells, W.M., Colchester, A., 
        Delp, S. (eds) Medical Image Computing and Computer-Assisted Intervention 
        — MICCAI’98. MICCAI 1998. Lecture Notes in Computer Science, vol 1496. 
        Springer, Berlin, Heidelberg.
        :DOI:`10.1007/BFb0056195`
    """
    results = np.empty(scales.shape + input.shape)
    out_image = np.empty(input.shape)
    directions = np.empty(scales.shape + input.shape)
    out_directions = np.empty(input.shape)

    #padding to catch and later cut off artifacts at image borders
    pad = int(np.max(scales)*2)
    pad_x = (pad, input.shape[0] + pad)
    pad_y = (pad, input.shape[1] + pad)
    input = np.pad(input, [pad_x, pad_y], mode='edge')

    for i, scale in enumerate(scales):
        eigenvals, eigenvects = _get_hessian_eigvals(input, scale, sorted_by_abs=True)

        norm = np.sqrt(eigenvals[0]**2 + eigenvals[1]**2)
        max_norm = np.max(norm)
    
        c = max_norm/2

        abs_eigen1 = np.abs(eigenvals[0])
        abs_eigen2 = np.abs(eigenvals[1]) + 1e-10 #avoid division by 0
        blob_ratios = abs_eigen1 / abs_eigen2
        result = np.where(eigenvals[1] > 0, 0, np.exp(-blob_ratios**2/2*blobness**2) * (1-np.exp(-norm**2/(2*c**2))))
        results[i] = result[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]
        if get_directions == True:
            directions[i] = _get_eigvect_directions(eigenvects)[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]
        
        print("Finished scale = %i"%(scale))

    out_image = np.max(results, axis=0)

    if get_directions == True:
        max_indices = np.argmax(results, axis=0)
        for i in range(scales.size):
            out_directions = np.where(max_indices == i, directions[i,:,:], out_directions)             
        return out_image, out_directions
    else:
        return out_image


def compute_meijering(input, scale, get_directions=False, sorted_by_abs=False):
    """
    Meijering Filter [1]_ for 2D images.

    Parameters
    ----------
    input : 2D ndarray
        Image that will be filtered.
    scale : int
        Expected structure widths.
    blobness : float
        Ratio at which blob-like structures will be included in the result.
        Low values will cause blob-like structures to fade.
        Accepts values between 0 and 1.
    get_directions : bool, optional
        Returns the direction of lowest curvature for each pixel in addition 
        to the filtered image.


    Returns
    -------
    out_image : 2D ndarray
        Filtered image.
    out_directions : 2D ndarray, optional
        Direction of lowest curvature for each pixel.

    References
    ----------
    .. [1] Meijering, E., Jacob, M., Sarria, J.-.-C.F., Steiner, P., Hirling, H. and 
    Unser, M. (2004), Design and validation of a tool for neurite tracing and 
    analysis in fluorescence microscopy images. Cytometry, 58A: 167-176. 
    :DOI:`10.1002/cyto.a.20022`
    """
    out_image = np.empty(input.shape)

    #padding to catch and later cut off artifacts at image borders
    pad = int(scale*2)
    pad_x = (pad, input.shape[0] + pad)
    pad_y = (pad, input.shape[1] + pad)
    input = np.pad(input, [pad_x, pad_y], mode='edge')

    eigenvals, eigenvects = _get_hessian_eigvals(input, scale)

    mod_eigens_1 = eigenvals[0] - eigenvals[1]/3
    mod_eigens_2 = eigenvals[1] - eigenvals[0]/3
    max_mod_eigens = np.where(np.abs(mod_eigens_1) > np.abs(mod_eigens_2), mod_eigens_1, mod_eigens_2)
    min_mod_eigen = np.min(np.minimum(mod_eigens_1, mod_eigens_2))
    out_image = np.where(max_mod_eigens < 0, max_mod_eigens/min_mod_eigen, 0)
    out_image = out_image[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]

    if get_directions == True:
        out_directions = _get_eigvect_directions(eigenvects)[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]
        return out_image, out_directions
    else:
        return out_image

def compute_rvr(input, scales, tau, get_directions=False, white_ridges=True):
    """
    Filter nach Jerman et al. [1]_ for 2D images.

    Parameters
    ----------
    input : 2D ndarray
        Image that will be filtered.
    scales : [int]
        List of expected structure widths.
    tau : float
        Parameter for calculation of normalized eigenvalues.
        High values will shrink filtered structures and low 
        values will expand filtered structures.
    get_directions : bool, optional
        Returns the direction of lowest curvature for each pixel in addition 
        to the filtered image.


    Returns
    -------
    out_image : 2D ndarray
        Filtered image.
    out_directions : 2D ndarray, optional
        Direction of lowest curvature for each pixel.

    References
    ----------
    .. [1] T. Jerman, F. Pernuš, B. Likar and Ž. Špiclin, "Enhancement of Vascular 
        Structures in 3D and 2D Angiographic Images," in IEEE Transactions on 
        Medical Imaging, vol. 35, no. 9, pp. 2107-2118, Sept. 2016.
        :DOI:`10.1109/TMI.2016.2550102`
    """
    results = np.empty(scales.shape + input.shape)
    out_image = np.empty(input.shape)
    directions = np.empty(scales.shape + input.shape)
    out_directions = np.empty(input.shape)

    #padding to catch and later cut off artifacts at image borders
    pad = int(np.max(scales)*2)
    pad_x = (pad, input.shape[0] + pad)
    pad_y = (pad, input.shape[1] + pad)
    input = np.pad(input, [pad_x, pad_y], mode='edge')

    for i, scale in enumerate(scales):
        eigenvals, eigenvects = _get_hessian_eigvals(input, scale, sorted_by_abs=True, white_ridges_rvr=white_ridges)

        eigen2 = eigenvals[1]

        threshold = tau * np.max(eigen2)
        
        regular_eigen = np.where(eigen2 <= threshold, threshold, 0)
        regular_eigen = np.where(0 < eigen2, 0, regular_eigen)
        regular_eigen = np.where(eigen2 > threshold, eigen2, regular_eigen)

        result =  np.where(eigen2 >= regular_eigen/2, 1, eigen2**2 * (regular_eigen - eigen2) * (3/(eigen2+regular_eigen))**2)
        result =  np.where(eigen2 <= 0, 0, result)
        result =  np.where(regular_eigen <= 0, 0, result)
        results[i] = result[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]
        if get_directions == True:
            directions[i] = _get_eigvect_directions(eigenvects)[pad_x[0]:pad_x[1],pad_y[0]:pad_y[1]]

        print("Finished sigma %i"%scale)
        
    out_image = np.max(results, axis=0)

    if get_directions == True:
        max_indices = np.argmax(results, axis=0)
        for i in range(scales.size):
            out_directions = np.where(max_indices == i, directions[i,:,:], out_directions)             
        return out_image, out_directions
    else:
        return out_image

def _get_hessian_eigvals(input, scale, sorted_by_abs=False, white_ridges_rvr=False, axis=0):
    """
    Calculates and preprocesses eigenvalues for Hesse-matrix based filters.

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
    hessian = hessian_matrix(input, sigma=scale)
    eigvals, eigvects = _symmetric_compute_eigenvalues(hessian)
    if white_ridges_rvr == True:
        eigvals = eigvals * -1

    if sorted_by_abs == True:
        # Create auxiliary array for indexing
        index = list(np.ix_(*[np.arange(i) for i in eigvals.shape]))

        # Get indices of abs sorted array
        index[axis] = np.abs(eigvals).argsort(axis)

        # Return abs sorted array
        return eigvals[tuple(index)], eigvects[tuple(index)]
    
    else: 
        return eigvals, eigvects
    
def sort_by_abs(input, scale, axis=0):
    """
    Calculates eigenvalues and sorts them by absolute value.

    Parameters
    ----------
    input : 2D ndarray
        Image that will be filtered.
    scale : int
        Expected structure width.

    Returns
    -------
    eigvals : ndarray
        Sorted eigenvalues of the input.
    """
    hessian = hessian_matrix(input, sigma=scale)
    eigvals, _ = _symmetric_compute_eigenvalues(hessian)

    # Create auxiliary array for indexing
    index = list(np.ix_(*[np.arange(i) for i in eigvals.shape]))

    # Get indices of abs sorted array
    index[axis] = np.abs(eigvals).argsort(axis)

    # Return abs sorted array
    return eigvals[tuple(index)]

def _get_eigvect_directions(eigvects):
    """Caculates the directions of the eigenvectors corresponding to the 
    eigenvalues with smallest absolute value."""
    cartesians = eigvects[0,:,:,:]
    theta = np.arctan2(cartesians[:,:,1],cartesians[:,:,0])
    theta = np.rad2deg(theta)
    theta = np.where(theta < 0, theta + 180, theta) 
    return theta

#Skimage functions, modified to also return eigenvectors

def _symmetric_compute_eigenvalues(S_elems):
    """Custom Variant of Scikit-image code.

    References
    ----------
    https://github.com/scikit-image/scikit-image/blob/v0.20.0/skimage/feature/corner.py#L383"""
    matrices = _symmetric_image(S_elems)
    # eigvalsh returns eigenvalues in increasing order. We want decreasing
    eigs, vects = np.linalg.eigh(matrices)
    eigs = eigs[..., ::-1]
    vects = vects[..., ::-1, :]
    # transpose eigenvalues to shape (2, image.shape) and eigenvectors to (2, image.shape, 2)
    leading_axes_eig = tuple(range(eigs.ndim - 1))
    leading_axes_vects = tuple(range(vects.ndim - 1))
    return np.transpose(eigs, (eigs.ndim - 1,) + leading_axes_eig), np.transpose(vects, (vects.ndim - 1,) + leading_axes_vects)
    
def _symmetric_image(S_elems):
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