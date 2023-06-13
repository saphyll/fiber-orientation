# Fiber-Orientations
This repository provides Python functions and classes for the detection and visual enhancement of curvilinear structures in greyscale images, automatic analysis of their direction and visualization of detected directions.

## Usage Example
 ```python
    steps = []
    steps.append(CLAHE(16, .2))
    #steps.append(IUWT(np.arange(3,5), 80))
    #steps.append(SOAGK(np.arange(10,21), [1.1], 180, invert=True))
    #steps.append(Frangi(np.arange(20,41), 0.2))
    steps.append(PC(8, np.arange(5,8), .5, 10, 'max'))
    pipeline = Pipeline(steps)

    dataset = Dataset('img/examples', 'results/datasets/examples')
    dataset.save_dataset_results(pipeline, 
                                save_orientation_histograms=True, 
                                save_colored_orientations=True, 
                                save_tiled_orientations=True, 
                                tile_num=16)
```