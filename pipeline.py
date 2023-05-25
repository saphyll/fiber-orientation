import matplotlib.pyplot as plt
import os
from pathlib import Path
import cv2

class Pipeline:
    def __init__(self, steps, save_path=None, auto_path=True) -> None:
        self.steps = steps
        self.save_path = save_path
        self.auto_path = auto_path
    
    def run(self, image_path, get_directions=False):
        image = cv2.imread(image_path, flags=cv2.IMREAD_ANYDEPTH)
        image_name = Path(image_path).stem
        for i, step in enumerate(self.steps):
            image, direction_map = step.compute(image, get_directions=get_directions)
            if i < len(self.steps)-1 and self.save_path != None and self.auto_path == True:
                self.save_path = self.save_path + step.get_descriptor() + "/"
            elif i == len(self.steps)-1:
                last_descriptor = step.get_descriptor()
        if self.save_path == None:
            return image, direction_map
        else:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            fig, ax = plt.subplots(figsize=(image.shape[0]/100, image.shape[1]/100))
            ax.imshow(image)
            ax.axis('off')
            plt.savefig("{}{}_{}.tif".format(self.save_path, image_name, last_descriptor))
            plt.close()
