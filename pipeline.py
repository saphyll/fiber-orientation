import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2
import numpy as np
import visuals
 
class Pipeline:
    def __init__(self, steps) -> None:
        """
        Infrastructure for image processing workflows and visualization creation.

        Parameters
        ----------
        steps : [filter objects]
            Sequence of Filters that will be applied by the pipeline.
        """
        self.steps = steps
        self.image_name = None
        self.result_image = None
        self.orientation_map = None
    
    def run(self, image_path, compute_orientations=False):
        """
        Computes pipeline results for the given image and optionally returns an 
        orientation map for the result. Results are stored as attributes of the 
        pipeline object.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        compute_orientations: bool, optional
            Specifies if an orientation map will be computed and returned.
        """
        image = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
        self.image_name = Path(image_path).stem
        for i, step in enumerate(self.steps):
            if compute_orientations == True:
                image, orientation_map = step.compute(image, get_directions=compute_orientations)
            else:
                image = step.compute(image, get_directions=compute_orientations)

        self.result_image = image
        if compute_orientations == True:
            self.orientation_map = orientation_map

    def save_processed_image(self, save_path, auto_path=True):
        """
        Saves the pipeline result for the last processed image at the 
        specified location. If no result exist, this will do nothing.

        Parameters
        ----------
        save_path : str
            Path to the save location.
        auto_path: bool, optional
            If true, the image will be saved in a folder structure representing 
            the processing steps and their parameters.
        """
        if self.result_image is None:
            pass

        for i, step in enumerate(self.steps):
            if i < len(self.steps)-1 and auto_path == True:
                save_path = save_path + "/" + step.get_descriptor()
            elif i == len(self.steps)-1:
                last_descriptor = step.get_descriptor()

        if not os.path.exists(save_path):
                os.makedirs(save_path)

        fig, ax = plt.subplots(figsize=(self.result_image.shape[0]/100, self.result_image.shape[1]/100))
        ax.imshow(self.result_image)
        ax.axis('off')
        plt.savefig("{}/processed_{}_{}.tif".format(save_path, self.image_name, last_descriptor))
        plt.close()

    def save_orientation_histograms(self, save_path, auto_path=True):
        """
        Saves a histogram for the orientation map of the last processed image at the 
        specified location. If no orientation map exists, this will do nothing.

        Parameters
        ----------
        save_path : str
            Path to the save location.
        auto_path: bool, optional
            If true, the image will be saved in a folder structure representing 
            the processing steps and their parameters.
        """
        if self.orientation_map is None:
            pass
        
        for i, step in enumerate(self.steps):
            if i < len(self.steps)-1 and auto_path == True:
                save_path = save_path + "/" + step.get_descriptor()
            elif i == len(self.steps)-1:
                last_descriptor = step.get_descriptor()

        if not os.path.exists(save_path):
                os.makedirs(save_path)

        orientation_histo, bins = visuals.get_orientation_histogram(self.result_image,
                                                                       self.orientation_map)
        orientation_histo = orientation_histo / self.orientation_map.size * 100

        fig, ax = plt.subplots(figsize=(18,10), dpi=300)
        ax.bar(bins, orientation_histo, width=.5)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        
        plt.savefig("{}/orientation_histograms_{}_{}.tif".format(save_path, self.image_name, last_descriptor))
        plt.close()

        csv_path = "{}/orientation_histograms_{}_{}.csv".format(save_path, self.image_name, last_descriptor)
        with open(csv_path, "a", newline="") as csvfile:
            csvfile.write("{},".format(self.image_name))
            orientation_histo = np.round(orientation_histo, 5)
            np.savetxt(csvfile, orientation_histo[np.newaxis,:], fmt='%3.5f', delimiter=",")

    def save_colored_orientations(self, save_path, auto_path=True):
        """
        Saves the last processed image with pixels colored corresponding to their 
        directions at the specified location. If no result image or orientation map exists, 
        this will do nothing.

        Parameters
        ----------
        save_path : str
            Path to the save location.
        auto_path: bool, optional
            If true, the image will be saved in a folder structure representing 
            the processing steps and their parameters.
        """
        if self.orientation_map is None or self.result_image is None:
            pass

        for i, step in enumerate(self.steps):
            if i < len(self.steps)-1 and auto_path == True:
                save_path = save_path + "/" + step.get_descriptor()
            elif i == len(self.steps)-1:
                last_descriptor = step.get_descriptor()

        if not os.path.exists(save_path):
                os.makedirs(save_path)

        black_image = np.zeros(self.result_image.shape)
        fig, ax = plt.subplots(figsize=(self.result_image.shape[0]/100, self.result_image.shape[1]/100))
        #ax.imshow(image, cmap=mpl.colormaps['gray'])
        ax.imshow(black_image, cmap=mpl.colormaps['gray'])
        ax.imshow(self.orientation_map, alpha=self.result_image, cmap=mpl.colormaps['hsv'])
        ax.axis('off')

        plt.savefig("{}/colored_{}_{}.tif".format(save_path, self.image_name, last_descriptor))
        plt.close()

    def save_tiled_orientations(self, tile_num, save_path, auto_path=True):
        """
        Saves the last processed result image with tiled direction visualisation. The image is divided 
        into the specified numbert of tiles in each direction. Each tile is colored corresponding to 
        the direction that occurs the most inside the tile. The spread of directions in the tile is shown
        as transparency, with low spread causing low transparency. Additionally, the most occuring 
        dirrection in each tile will be shown as a white line in the center of the tile.

        Parameters
        ----------
        tile_num : int
            Number of tiles in each direction that the image will be divided into.
        save_path : str
            Path to the save location.
        auto_path: bool, optional
            If true, the image will be saved in a folder structure representing 
            the processing steps and their parameters.
        """
        if self.orientation_map is None or self.result_image is None:
            raise ValueError("Result image or orientation map is missing. Did you use Pipeline.run() with compute_orientations=True?")

        for i, step in enumerate(self.steps):
            if i < len(self.steps)-1 and auto_path == True:
                save_path = save_path + "/" + step.get_descriptor()
            elif i == len(self.steps)-1:
                last_descriptor = step.get_descriptor()

        if not os.path.exists(save_path):
                os.makedirs(save_path)

        tile_shape = np.array([self.result_image.shape[0]//tile_num + 1, self.result_image.shape[1]//tile_num + 1])
        color_tiles = np.zeros(self.result_image.shape)
        entropy_tiles = np.zeros(self.result_image.shape)
        bar_tiles = np.zeros(self.result_image.shape) 

        for i in range(tile_num):
            for j in range(tile_num):
                start_x = i * tile_shape[0]
                start_y = j * tile_shape[1]
                end_x = np.clip(start_x+tile_shape[0], None, self.result_image.shape[0])
                end_y = np.clip(start_y+tile_shape[1], None, self.result_image.shape[1])
                direction_tile = self.orientation_map[start_x:end_x,start_y:end_y]
                image_tile = self.result_image[start_x:end_x,start_y:end_y]
                primary_direction, entropy = visuals.get_primary_orientation(image_tile, direction_tile)
                color_tiles[start_x:end_x,start_y:end_y] = primary_direction
                entropy_tiles[start_x:end_x,start_y:end_y] = entropy
                bar_tiles[start_x:end_x,start_y:end_y] = visuals.get_orientation_bar(direction_tile.shape, primary_direction)

        #set discarded entropy values to maximum entropy
        entropy_tiles = np.where(entropy_tiles == -1, np.max(entropy_tiles), entropy_tiles)
        
        #normalize entropy to range [0,1]
        entropy_tiles = entropy_tiles - np.min(entropy_tiles)
        entropy_tiles = entropy_tiles / np.max(entropy_tiles)

        fig, ax = plt.subplots(figsize=(self.result_image.shape[0]/100, self.result_image.shape[1]/100))
        fig.subplots_adjust(0,0,1,1)
        ax.axis('off')
        #ax.imshow(black_image, cmap=mpl.colormaps['gray'])
        ax.imshow(self.result_image, cmap=mpl.colormaps['gray'])
        ax.imshow(color_tiles, alpha=1-entropy_tiles, cmap=mpl.colormaps['hsv'])
        ax.imshow(bar_tiles, alpha=bar_tiles*(1-entropy_tiles), cmap=mpl.colormaps['gray'])

        plt.box(False)
        plt.savefig("{}/tiled_{}_{}.tif".format(save_path, self.image_name, last_descriptor))
        plt.close()

    def show_interactive_orientations(self, orientation_num):
        """
        Shows an interactive view of the last processed result image which allows to isolate specific directions 
        in the image.
        If no result image or orientation map exists, this will do nothing.

        Parameters
        ----------
        orientation_num : int
            Number of orientations that can be isolated.
        """
        if self.orientation_map is None or self.result_image is None:
            raise ValueError("Result image or orientation map is missing. Did you use Pipeline.run() with compute_orientations=True?")

        visuals.interactive_orientation_segmentation(self.result_image, self.orientation_map, orientation_num)

    def get_dataset_entropies(self, tile_num):
        """
        Calculates the highest and lowest entropies of the orientation histograms in the tiled image.

        Parameters
        ----------
        tile_num : int
            Number of tiles in each direction that the image will be divided into.

        Returns
        -------
        min_entropy : float
            Lowest calculated entropy.
        max_entropy : float
            Highest calculated entropy.
        """
        tile_shape = np.array([self.result_image.shape[0]//tile_num + 1, self.result_image.shape[1]//tile_num + 1])
        color_tiles = np.zeros(self.result_image.shape)
        entropy_tiles = np.zeros(self.result_image.shape)
        bar_tiles = np.zeros(self.result_image.shape) 

        for i in range(tile_num):
            for j in range(tile_num):
                start_x = i * tile_shape[0]
                start_y = j * tile_shape[1]
                end_x = np.clip(start_x+tile_shape[0], None, self.result_image.shape[0])
                end_y = np.clip(start_y+tile_shape[1], None, self.result_image.shape[1])
                direction_tile = self.orientation_map[start_x:end_x,start_y:end_y]
                image_tile = self.result_image[start_x:end_x,start_y:end_y]
                _ , entropy = visuals.get_primary_orientation(image_tile, direction_tile)
                entropy_tiles[start_x:end_x,start_y:end_y] = entropy
            
        #set discarded entropy values to maximum entropy
        entropy_tiles = np.where(entropy_tiles == -1, np.max(entropy_tiles), entropy_tiles)

        return np.min(entropy_tiles), np.max(entropy_tiles)
    
    def save_results_csv(self, save_path):
        """
        Saves the filtered image and the corresponding orientation map as CSV-files.

        Parameters
        ----------
        save_path: int
            Path to the location where the CSV-files will be saved.
        """
        csv_path_filtered_image = "{}/csv/{}_filtered.csv".format(save_path, self.image_name)
        csv_path_orientations = "{}/csv/{}_orientations.csv".format(save_path, self.image_name)
        
        if not os.path.exists("{}/csv/".format(save_path)):
                os.makedirs("{}/csv/".format(save_path))
        
        with open(csv_path_filtered_image, "w", newline="") as csvfile:
            np.savetxt(csvfile, self.result_image, fmt='%3.5f', delimiter=";")
        with open(csv_path_orientations, "w", newline="") as csvfile:
            np.savetxt(csvfile, self.orientation_map, fmt='%3.5f', delimiter=";")