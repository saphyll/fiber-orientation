import numpy as np
from enum import Enum

class lhe_type(Enum):
    LHE = 0
    CLAHE = 1
    CLHE = 2

def equalize_histo(input):
    """
    Performs global histogram equalization on a 16-bit grayscale image.

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

    return output.reshape(input.shape), cdf, bins

def compute_lhe(input, area_num):
    """
    Performs simple Adaptive Histogram Equalization on a 16-bit grayscale image.

    Parameters
    ----------
    input : 2D ndarray
        Image that will be equalized.
    area_num : int
        Number of sample points/areas for interpolation in x-/y-direction 
        respectively. Odd numbers will be converted to the next lower even 
        number.

    Returns
    -------
    output : 2D ndarray
        Equalized image.
    """
    return _lhe_variants(input, area_num, type=lhe_type.LHE)

def compute_clahe(input, area_num, beta):
    """
    Performs Contrast Limited Adaptive Histogram Equalization on a 16-bit grayscale image.

    Parameters
    ----------
    input : 2D ndarray
        Image that will be equalized.
    area_num : int
        Number of sample points/areas for interpolation in x-/y-direction 
        respectively. Odd numbers will be converted to the next lower even 
        number.
    beta : float 
        Value at which the normalized Histogram is clipped. 
        Takes values between 0 and 1.

    Returns
    -------
    output : 2D ndarray
        Equalized image.
    """
    return _lhe_variants(input, area_num, type=lhe_type.CLHE, beta=beta)

def compute_clhe(input, area_num, alpha):
    """
    Performs Constrained Local Histogram Equalization on a 16-bit grayscale image.

    Parameters
    ----------
    input : 2D ndarray
        Image that will be equalized.
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

    Returns
    -------
    output : 2D ndarray
        Equalized image.
    """
    return _lhe_variants(input, area_num, type=lhe_type.CLHE, alpha=alpha)

def _lhe_variants(input, area_num, type=lhe_type.LHE, alpha=0.5, beta=0.8):
    """
    Computes one of these three variants of LHE:
    - Standard LHE (TODO:Source)
    - CLHE (TODO:Source)
    - CLAHE (TODO:Source)


    Parameters
    ----------
    input : 2D ndarray
        Image that will be equalized.
    area_num : int
        Number of sample points/areas for interpolation in x-/y-direction 
        respectively. Odd numbers will be converted to the next lower even 
        number.
    type: Enum lhe_type
        Variant of LHE that will be computed.
    alpha : float, optional
        Specifies the influence of the local histogram and the histogram 
        of the remaining image on the result. Values below 0.5 will 
        emphasize the local histogram and values above 0.5 will emphasize 
        the histogram of the remaining image. 
        Takes values between 0 and 1.
    beta : float, optional
        Value at which the normalized Histogram is clipped. 
        Takes values between 0 and 1.
    """
    output = np.empty(input.shape)
    if area_num%2 != 0:
        area_num = area_num - 1
    area_size = np.array([input.shape[0]//area_num + 1, input.shape[1]//area_num + 1])

    global_histo = None
    if type == lhe_type.CLHE:
        global_histo, _ = np.histogram(input, bins=2**16, range=(0,2**16))

    mapped_images = np.empty((2,2)+(input.shape))
    for i in range(area_num):
        k = i % 2
        for j in range(area_num):
            l = j % 2
            #define histogram regions and compute histograms
            area_start_x = i * area_size[0]
            area_start_y = j * area_size[1]
            area_end_x = np.clip(area_start_x+area_size[0], None, input.shape[0])
            area_end_y = np.clip(area_start_y+area_size[1], None, input.shape[1])
            area_mid_x = area_start_x + area_size[0]//2
            area_mid_y = area_start_y + area_size[1]//2
            histo_area = input[area_start_x:area_end_x,area_start_y:area_end_y]
            if type == lhe_type.LHE:
                histo, bins = np.histogram(histo_area, bins=2**16, range=(0,2**16)) #normalize
                histo = histo / (histo_area.shape[0] * histo_area.shape[1])
                bins = bins[:2**16]
            if type == lhe_type.CLHE:
                local_histo, bins = np.histogram(histo_area, bins=2**16, range=(0,2**16))
                bins = bins[:2**16]
                rest_histo = global_histo - local_histo
                local_histo = local_histo / histo_area.size #normalize
                rest_histo = rest_histo  / (input.size - histo_area.size) #normalize
                histo = alpha * local_histo + (1 - alpha) * rest_histo   
            if type == lhe_type.CLAHE:
                histo, bins = np.histogram(histo_area, bins=2**16, range=(0,2**16)) #normalize
                histo = histo / (histo_area.shape[0] * histo_area.shape[1])
                bins = bins[:2**16]
                #redistribute histogram counts that exceed beta
                excess = np.sum(np.where(histo > beta, histo - beta, 0))
                redist_amount = excess / np.count_nonzero(histo)
                histo = np.clip(histo, None, beta)
                for i in range(2**16):
                    if histo[i] > 0:
                        if histo[i] < beta - redist_amount:
                            excess = excess - redist_amount
                            histo[i] = histo[i] + redist_amount
                        elif histo[i] < beta:
                            excess = excess - (beta - histo[i])
                            histo[i] = beta
                histo = np.where(histo != 0, histo + excess/np.count_nonzero(histo), histo)
            
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

    triangle_x = np.array(list(triangle(area_size[0]*2, input.shape[0])))
    triangle_y = np.array(list(triangle(area_size[1]*2, input.shape[1])))
    weight_function = triangle_x[:, np.newaxis] *  triangle_y[:, np.newaxis].T
    
    #interpolate mapped images
    mapped_images = mapped_images.reshape(4,input.shape[0], input.shape[1])
    offset_directions = [[1,1],[1,0],[0,1],[0,0]]
    out_weights = []
    for i in range(4):
        mapped_image = mapped_images[i]
        weight_offsets = offset_directions[i] * (area_size - area_size%2)
        padded_weights = np.pad(weight_function, (area_size[0]//2,area_size[1]//2))
        weights = padded_weights[weight_offsets[0]:input.shape[0]+weight_offsets[0], weight_offsets[1]:input.shape[1]+weight_offsets[1]]
        out_weights.append(weights)
        output = output + weights * mapped_image

    return output, mapped_images[1], out_weights[1]

def triangle(period, list_length):
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