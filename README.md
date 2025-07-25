# ELSED+GC: Enhanced Line SEgment Drawing + Gradient Classification for Soccer Field Lines Detection

This repository adds a gradient classification to the [**ELSED: Enhanced Line SEgment Drawing**](https://doi.org/10.1016/j.patcog.2022.108619) algorithm, which allows us to classify line segments based on their color transitions in the RGB color space. Our implementation aims at classifying if the detected line segments belong to a soccer field or not, but it can be modified to address other types of line segments.

![Result example image](images/result.png)

## Dependencies (TO DO)

## Running

### Compiling ELSED

The code can be compiled with Cmake:

```shell script
mkdir build && cd build
cmake .. && make
```


### Using ELSED+GC from python

Here is an example code on how to run ELSED+GC in a single image and highlight lines classification on screen. Lines classified as "not a field feature" are painted in black.

```python
import cv2
from elsed_analyzer import SegmentsAnalyzer

analyzer = SegmentsAnalyzer(draw_on_frames=True)

img_path = 'images/test_image.jpg'

img = cv2.imread(img_path)

analyzer.detect(img)

cv2.imshow('detections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Note that, the default thresholds in the [SegmentsAnalyzer](https://github.com/jgocm/ELSED-SSL/blob/7ad8fd7f30dbffbc468e7423627862db6ea3e17d/elsed_analyzer.py#L9) class are adjusted for our soccer field. To apply ELSED+GC to a new envinroment, the thresholds should be readjusted accordingly. This adjusment can be done manually or by following our thresholds training pipeline.

## Available Datasets and Thresholds (TO DO)


## Thresholds Training Pipeline
The procedure to adjust thresholds to a new soccer field consists of 4 main steps:
1. Collecting images
2. Annotating training data
3. Traning thresholds with PSO
4. Loading thresholds from the numpy file

### Collecting images
Take pictures of your environment from multiple perspectives to ensure all the lines were seen by different points of view.

Another way to collect it is by recording a video from the robot's camera and extracting frames from it. The [select_frames_from_video.py](https://github.com/jgocm/ELSED-SSL/blob/62eccad56d3c2046fa6a192f25bea43ca581923f/select_frames_from_video.py) script helps saving frames from a video.

### Annotating training data
The [annotate_trainings.py](https://github.com/jgocm/ELSED-SSL/blob/62eccad56d3c2046fa6a192f25bea43ca581923f/annotate_trainings.py) script loads all images from a given folder, runs ELSED to detect all line segments on it, and lets the user annotate the segments' labels manually one by one, for each image. Each segment can be labeled as: 'is_field_boundary', 'is_field_marking', or 'is_not_a_field_feature'.

These annotations are saved in a .csv file that contains the following information about each line segment:
- image path
- coordinates of the endpoints
- horizontal and vertical BGR gradients
- segment length
- ground truth label

The script provides the following keyboard commands for the user to make the annotations:
- q: quits without saving
- s: skips the current image and saves the current annotations to the csv file
- r: removes the last annotation
- d: skip/discard the current line segment
- b: annotate the current line segment as a field boundary
- m: annotate the current line segment as a field marking

Saving the annotations will append them to the current csv file, which allows the user to stop annotating before finishing all the images and continue later on.

### Training with PSO
The [train_and_evaluate_marking_thresholds.py](https://github.com/jgocm/ELSED-SSL/blob/62eccad56d3c2046fa6a192f25bea43ca581923f/train_and_evaluate_marking_thresholds.py) script runs a PSO to try to find the best thresholds for the given annotations file. It is set to run multiple trainings with different percentages of the training data. It saves the resulting thresholds for each training and, at the end, runs them on a test set to evaluate their performances.

Thresholds files are saved in .npy format and can be loaded with a simple ```np.load(PATH_TO_FILE)```.

Note that there are two scripts for training thresholds, one for field markings and another for field boundaries. The markings are classified based on a similarity to a green-to-white transition (colors of the field lines and the field itself), while the boundaries compute similarities to a green-to-black transition (the color transition for boundaries in the Small Size League field). Therefore, gradient classification for boundaries might not work in fields that have other colors in their borders.

### Loading thresholds
```python
import cv2
import numpy as np
from elsed_analyzer import SegmentsAnalyzer

PATH_TO_THRESHOLDS = 'trainings/humanoid-kid/marking_thresholds_40.npy'
markings_thresholds = np.load(PATH_TO_THRESHOLDS)

analyzer = SegmentsAnalyzer(marking_thresholds=markings_thresholds, 
                            draw_on_frames=True)

img_path = 'images/test_image.jpg'

img = cv2.imread(img_path)

analyzer.detect(img)

cv2.imshow('detections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

## Cite

### Efficient Lines Detection for Robot Soccer 
Published at RoboCup 2025, still no DOI available

### [**ELSED: Enhanced Line SEgment Drawing**](https://doi.org/10.1016/j.patcog.2022.108619)
```bibtex
@article{suarez2022elsed,
      title={ELSED: Enhanced Line SEgment Drawing}, 
      author={Iago Suárez and José M. Buenaposada and Luis Baumela},
      journal = {Pattern Recognition},
      volume = {127},
      pages = {108619},
      year = {2022},
      issn = {0031-3203},
      doi = {https://doi.org/10.1016/j.patcog.2022.108619},
      url = {https://www.sciencedirect.com/science/article/pii/S0031320322001005}
}
```
