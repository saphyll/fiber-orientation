import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import draw
from skimage.transform import rotate

def _isolate_direction(image, pixel_orients, orientation, deviation):
    """
    Isolates the pixels in an image, whose orientation is part of the values covered by 
    the given range of orientations. The range is defined by a center orientation and 
    a maximum deviation from this value.

    Parameters
    ----------
    image : 2D ndarray
        Image whose intensities will be shown in the result.
    pixel_orients : 2D ndarray
        Map of orientations corresponding to the image.
    orientation : int
        Target orientation that will be shown.
    deviation : int
        Range of orientations around the center value that will be shown in addition to the latter.
    
    Returns
    -------
    isolated_pixels : 2D ndarray
        Intensities of the input image where the pixel orientations are in the defined orientation range.
    """
    isolated_pixels = np.where(orientation - deviation < np.abs(pixel_orients), image, 0)
    isolated_pixels = np.where(pixel_orients < orientation + deviation, isolated_pixels, 0)

    isolated_wraparound = np.zeros(image.shape)
    if orientation - deviation < 0:
        low_wraparound = orientation - deviation + 180
        isolated_wraparound = np.where(low_wraparound < np.abs(pixel_orients), image, 0)
        isolated_pixels = isolated_pixels + isolated_wraparound

    if orientation + deviation > 180:
        high_wraparound = orientation + deviation - 180
        isolated_wraparound = np.where(high_wraparound > np.abs(pixel_orients), image, 0)
        isolated_pixels = isolated_pixels + isolated_wraparound

    return isolated_pixels

def interactive_orientation_segmentation(image, pixel_orients, orientation_num):
    """
    Interactive plot for the isolation of orientations in a filtered image.

    Parameters
    ----------
    image : 2D ndarray
        Image whose intensities will be shown in the result.
    pixel_orients : 2D ndarray
        Map of orientations corresponding to the image.
    orientation_num : int
        Number of separate orientations that can be isolated.
    """
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(image)
    fig.subplots_adjust(bottom=0.25)
    ax_deviation = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    ax_orientation = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    img = axs[1].imshow(_isolate_direction(image, pixel_orients, 0, 10))

    # create the sliders
    s_deviation = Slider(
        ax_deviation, "Deviation in °", 0, 45,
        valinit=10, valstep=1,
        color="green"
    )

    s_orient = Slider(
        ax_orientation, "Orientation in °", 0, 180,
        valinit=0, valstep=180/orientation_num,
        color="green"
    )

    def update(val):
        orientation = s_orient.val
        deviation = s_deviation.val
        data = _isolate_direction(image, pixel_orients, orientation, deviation)
        img.set_data(data)
        fig.canvas.draw_idle()

    s_deviation.on_changed(update)
    s_orient.on_changed(update)

    plt.show()
    plt.close()

def get_orientation_histogram(image, orientation_map):
    """
    Calculates an orientation histogram for the given orientation map and image.

    Parameters
    ----------
    image : 2D ndarray
        Image whose intensities will be shown in the result.
    orientation_map : 2D ndarray
        Map of orientations corresponding to the image.

    Returns
    -------
    orientation_histo : 1D ndarray
            Amount of existing values per bin.
    bins : 1D ndarray
        Orientations from 0 to 179° as ints.
    """
    #assign background pixels to an extra bin
    #assign background pixels to an extra bin
    direction_map = np.round(direction_map)
    direction_map = np.where(image < 0.1, 181, direction_map)
    orientation_histo, bins = np.histogram(direction_map, 181)
    #remove background pixels from histogram
    bins = bins[:bins.shape[0]-2]
    orientation_histo = orientation_histo[:orientation_histo.shape[0]-1]
    return orientation_histo, bins
    

def get_primary_orientation(image, orientation_map):
    """
    Calculates the primary orientation and entropy for the given orientation map and image.

    Parameters
    ----------
    image : 2D ndarray
        Image whose intensities will be shown in the result.
    orientation_map : 2D ndarray
        Map of orientations corresponding to the image.

    Returns
    -------
    primary_orientation : 1D ndarray
        Orientation that can be found most often in the orientation map.
    entropy : 1D ndarray
        Measure of orientation spread across the orientation map.
    """
    orientation_histo, bins = get_orientation_histogram(image, orientation_map)
    
    primary_orientation = bins[np.argmax(orientation_histo)]
    if np.max(image) < .1:
        entropy = -1
    else:
        orientation_histo = orientation_histo/orientation_map.size
        orientation_histo = orientation_histo[orientation_histo != 0]
        entropy = -1 * np.sum(np.log2(orientation_histo) * orientation_histo)

    return primary_orientation, entropy

def get_orientation_bar(area_shape, direction):
    """
    Calculates a rectangle shape that is centered in the given area and rotated to 
    represent the given direction.

    Parameters
    ----------
    area_shape : (int, int)
        Shape of the area that will contain the rectangle.
    direction : int
        Direction that will be represented, given in degrees.

    Returns
    -------
    area : 2D ndarray
        Area containing the rotated rectangle.
    """
    square_width = min(area_shape)
    area = np.zeros((square_width, square_width))
    start = (square_width//2 - square_width//16, square_width//8)
    extent = (square_width//8, square_width - square_width//8*2)

    rr, cc = draw.rectangle(start, extent=extent)
    area[rr, cc] = 1
    area = rotate(area, direction)

    if area_shape[0] != area_shape[1]:
        axis = np.argmax(area_shape)
        pad = max(area_shape) - min(area_shape) 
        pad_start = pad//2
        pad_end = pad - pad_start
        if axis == 0:   
            area = np.pad(area, [(pad_start,pad_end),(0,0)])
        else:
            area = np.pad(area, [(0,0),(pad_start,pad_end)])
        return area
    else:
        return area