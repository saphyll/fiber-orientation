import glob
import numpy as np
from pathlib import Path
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import visuals
import pipeline


class Dataset:
    def __init__(self, data_path, save_path) -> None:
        self.data_path = data_path
        self.save_path = save_path

    def save_dataset_results(self, 
                             pipeline : pipeline.Pipeline,
                             save_orientation_histograms=False,
                             save_colored_orientations=False,
                             save_tiled_orientations=False,
                             tile_num=16):
        """
        Computes filter results of the given processing pipeline for all TIF-files in the 
        source directory of this dataset. Result images and optionally other visualizations are saved at 
        the given location.

        Parameters
        ----------
        pipeline : Pipeline object
            Processing pipeline that will be applied to dataset images.
        save_orientation_histograms : bool, optional
            If true, orientation histograms are saved as images.
        save_colored_orientations : bool, optional
            If true, images of the filter results colored by pixel orientations are saved.
        save_tiled_orientations : bool, optional
            If true, images of primary orientations per tile are saved. The entropies used for 
            the visualization are normalized across the whole dataset.
        tile_num : int, optioanl
            Number of tiles, in x- and y- direction respectively, for the visualization of 
            primary orientations.
        """
        #self.save_results_csv(pipeline)
        #self.save_results_tif()
        if save_orientation_histograms:
            self.save_orientation_histograms()
        if save_colored_orientations:
            self.save_colored_orientations()
        if save_tiled_orientations:
            self.save_tiled_orientations(tile_num)
        return
    
    def save_results_csv(self, pipeline : pipeline.Pipeline):
        """
        Computes filter results of the given processing pipeline for all TIF-files in the 
        source directory of this dataset. Resuls are saved as CSV-files at the save location 
        of this dataset.

        Parameters
        ----------
        pipeline : Pipeline object
            Processing pipeline that will be applied to dataset images.
        """
        for image_path in glob.iglob(self.data_path + "/*.tif"):
            pipeline.run(image_path, compute_orientations=True)
            pipeline.save_results_csv(self.save_path)

    def save_results_tif(self):
        """
        Saves filter results from CSV-files as TIF-files.
        """
        for image_path in glob.iglob(self.data_path + "/*.tif"):
            image_name = Path(image_path).stem
            csv_path_filtered_image = "{}/csv/{}_filtered.csv".format(self.save_path, image_name)
            filtered_image = np.loadtxt(csv_path_filtered_image , delimiter=";")

            save_path = "{}/processed_images/".format(self.save_path)
            if not os.path.exists(save_path):
                    os.makedirs(save_path)

            fig, ax = plt.subplots(figsize=(filtered_image.shape[0]/100, filtered_image.shape[1]/100))
            ax.imshow(filtered_image)
            ax.axis('off')
            plt.savefig("{}/processed_images/{}.tif".format(self.save_path, image_name))
            plt.close()

    def save_orientation_histograms(self):
        """
        Saves orientation histograms for results in CSV-files.
        """
        for image_path in glob.iglob(self.data_path + "/*.tif"):
            image_name = Path(image_path).stem
            csv_path_filtered_image = "{}/csv/{}_filtered.csv".format(self.save_path, image_name)
            csv_path_orientations = "{}/csv/{}_orientations.csv".format(self.save_path, image_name)
            filtered_image = np.loadtxt(csv_path_filtered_image , delimiter=";")
            orientations = np.loadtxt(csv_path_orientations , delimiter=";")

            save_path = "{}/orientation_histograms/".format(self.save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            orientation_histo, bins = visuals.get_orientation_histogram(filtered_image,
                                                                        orientations)
            orientation_histo = orientation_histo / orientations.size * 100

            fig, ax = plt.subplots(figsize=(18,10), dpi=300)
            ax.bar(bins, orientation_histo, width=.5)
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
            
            plt.savefig("{}/{}.tif".format(save_path, image_name))
            plt.close()

            csv_path = "{}/orientation_histogram_{}.csv".format(save_path, image_name)
            with open(csv_path, "a", newline="") as csvfile:
                csvfile.write("{},".format(image_name))
                orientation_histo = np.round(orientation_histo, 5)
                np.savetxt(csvfile, orientation_histo[np.newaxis,:], fmt='%3.5f', delimiter=",")

    def save_colored_orientations(self):
        """Saves filtered images colored by pixel orientations."""
        for image_path in glob.iglob(self.data_path + "/*.tif"):
            image_name = Path(image_path).stem
            csv_path_filtered_image = "{}/csv/{}_filtered.csv".format(self.save_path, image_name)
            csv_path_orientations = "{}/csv/{}_orientations.csv".format(self.save_path, image_name)
            filtered_image = np.loadtxt(csv_path_filtered_image , delimiter=";")
            orientations = np.loadtxt(csv_path_orientations , delimiter=";")

            save_path = "{}/colored_orientations/".format(self.save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            black_image = np.zeros(filtered_image.shape)
            fig, ax = plt.subplots(figsize=(filtered_image.shape[0]/100, filtered_image.shape[1]/100))
            fig.subplots_adjust(0,0,1,1)
            #ax.imshow(image, cmap=mpl.colormaps['gray'])
            ax.imshow(black_image, cmap=mpl.colormaps['gray'])
            ax.imshow(orientations, alpha=filtered_image, cmap=mpl.colormaps['hsv'])
            ax.axis('off')

            #plt.box(False)
            plt.savefig("{}/{}_colored.tif".format(save_path, image_name), bbox_inches = 'tight', pad_inches = 0)
            plt.close()

    def save_tiled_orientations(self, tile_num):
        """Saves primary orientations per image tile. The entropies used for 
            the visualization are normalized across the whole dataset.
        
        Parameters
        ----------
        tile_num : int
            Number of tiles, in x- and y- direction respectively, for the visualization of 
            primary orientations."""
        min_entropies = []
        max_entropies = []
        for image_path in glob.iglob(self.data_path + "/*.tif"):
            image_name = Path(image_path).stem
            csv_path_filtered_image = "{}/csv/{}_filtered.csv".format(self.save_path, image_name)
            csv_path_orientations = "{}/csv/{}_orientations.csv".format(self.save_path, image_name)
            filtered_image = np.loadtxt(csv_path_filtered_image , delimiter=";")
            orientations = np.loadtxt(csv_path_orientations , delimiter=";")
            min_entropy, max_entropy = self._get_image_entropies(tile_num, filtered_image, orientations)
            min_entropies.append(min_entropy)
            max_entropies.append(max_entropy)
        dataset_entropies = (min(min_entropies), max(max_entropies))
        
        for image_path in glob.iglob(self.data_path + "/*.tif"):
            image_name = Path(image_path).stem
            csv_path_filtered_image = "{}/csv/{}_filtered.csv".format(self.save_path, image_name)
            csv_path_orientations = "{}/csv/{}_orientations.csv".format(self.save_path, image_name)
            filtered_image = np.loadtxt(csv_path_filtered_image , delimiter=";")
            orientations = np.loadtxt(csv_path_orientations , delimiter=";")

            save_path = "{}/tiled_orientations/".format(self.save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            tile_shape = np.array([filtered_image.shape[0]//tile_num + 1, filtered_image.shape[1]//tile_num + 1])
            color_tiles = np.zeros(filtered_image.shape)
            entropy_tiles = np.zeros(filtered_image.shape)
            bar_tiles = np.zeros(filtered_image.shape) 

            for i in range(tile_num):
                for j in range(tile_num):
                    start_x = i * tile_shape[0]
                    start_y = j * tile_shape[1]
                    end_x = np.clip(start_x+tile_shape[0], None, filtered_image.shape[0])
                    end_y = np.clip(start_y+tile_shape[1], None, filtered_image.shape[1])
                    direction_tile = orientations[start_x:end_x,start_y:end_y]
                    image_tile = filtered_image[start_x:end_x,start_y:end_y]
                    primary_direction, entropy = visuals.get_primary_orientation(image_tile, direction_tile)
                    color_tiles[start_x:end_x,start_y:end_y] = primary_direction
                    entropy_tiles[start_x:end_x,start_y:end_y] = entropy
                    bar_tiles[start_x:end_x,start_y:end_y] = visuals.get_orientation_bar(direction_tile.shape, primary_direction)

            #set discarded entropy values to maximum entropy
            entropy_tiles = np.where(entropy_tiles == -1, dataset_entropies[1], entropy_tiles)

            #normalize entropy to range [0,1]
            entropy_tiles = entropy_tiles - dataset_entropies[0]
            entropy_tiles = entropy_tiles / (dataset_entropies[1] - dataset_entropies[0])
            #workaround for float precision issues that cause some values to be slightly outside of interval [0,1]
            entropy_tiles = np.clip(entropy_tiles, 0, 1)

            fig, ax = plt.subplots(figsize=(filtered_image.shape[0]/100, filtered_image.shape[1]/100))
            fig.subplots_adjust(0,0,1,1)
            ax.axis('off')
            #ax.imshow(black_image, cmap=mpl.colormaps['gray'])
            ax.imshow(filtered_image, cmap=mpl.colormaps['gray'])
            ax.imshow(color_tiles, alpha=1-entropy_tiles, cmap=mpl.colormaps['hsv'])
            ax.imshow(bar_tiles, alpha=bar_tiles*(1-entropy_tiles), cmap=mpl.colormaps['gray'])

            plt.box(False)
            plt.savefig("{}/{}_tiled.tif".format(save_path, image_name), bbox_inches = 'tight', pad_inches = 0)
            plt.close()

    def _get_image_entropies(self, tile_num, result_image, orientation_map):
        """Calculates minimum and maximum entropies for a tiled image."""
        tile_shape = np.array([result_image.shape[0]//tile_num + 1, result_image.shape[1]//tile_num + 1])
        color_tiles = np.zeros(result_image.shape)
        entropy_tiles = np.zeros(result_image.shape)
        bar_tiles = np.zeros(result_image.shape) 

        for i in range(tile_num):
            for j in range(tile_num):
                start_x = i * tile_shape[0]
                start_y = j * tile_shape[1]
                end_x = np.clip(start_x+tile_shape[0], None, result_image.shape[0])
                end_y = np.clip(start_y+tile_shape[1], None, result_image.shape[1])
                direction_tile = orientation_map[start_x:end_x,start_y:end_y]
                image_tile = result_image[start_x:end_x,start_y:end_y]
                _, entropy = visuals.get_primary_orientation(image_tile, direction_tile)
                entropy_tiles[start_x:end_x,start_y:end_y] = entropy
            
        #set discarded entropy values to maximum entropy
        entropy_tiles = np.where(entropy_tiles == -1, np.max(entropy_tiles), entropy_tiles)

        return np.min(entropy_tiles), np.max(entropy_tiles)