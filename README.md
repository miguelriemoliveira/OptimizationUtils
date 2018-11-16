# OptimizationUtils
A set of utilities for using the python scipy optimizer functions

# Usage

There are several examples. Here is how to launch them:

### Color Correction using an OC dataset

Uses the OCDatasetLoader to load an OC dataset and runs a color balancing optimization using images from the cameras.

```bash
test/color_balancing_oc_dataset.py -p ~/datasets/red_book2/dataset/ -m ~/datasets/red_book2/dataset/1528188687058_simplified_decimated.obj -i ~/datasets/red_book2/dataset/calibrations/camera.yaml -si 5
```

### Camera pose optimization using an OC dataset

Uses the OCDatasetLoader to load an OC dataset and runs a camera pose optimization.

```bash
test/camera_pose_oc_dataset.py -p ~/datasets/red_book_aruco/dataset/ -m ~/datasets/red_book_aruco/dataset/1528885039597.obj -i ~/datasets/red_book_aruco/dataset/calibrations/camera.yaml -si 15
```
