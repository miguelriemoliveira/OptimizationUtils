# OptimizationUtils
A set of utilities for using the python scipy optimizer functions

# Installation

To install you must add the path of this python module to your PYTHON_PATH environment variable. You may run this script from the directory where your script is.

```bash
bash install.sh
```

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
test/camera_pose_oc_dataset.py -p ~/datasets/red_book_aruco/dataset/ -m ~/datasets/red_book_aruco/dataset/1528885039597.obj -i ~/datasets/red_book_aruco/dataset/calibrations/camera.yaml -ms 0.082 -si 15
```

to view the aruco detections run:

```bash
test/camera_pose_oc_dataset.py -p ~/datasets/red_book_aruco/dataset/ -m ~/datasets/red_book_aruco/dataset/1528885039597.obj -i ~/datasets/red_book_aruco/dataset/calibrations/camera.yaml -ms 0.082 -vad -va3d -si 15
```

### Pose and color optimization using an OC dataset

Uses the OCDatasetLoader to load an OC dataset and runs a camera pose plus camera color optimization.

```bash
test/pose_and_color_oc_dataset.py -p ~/datasets/red_book_aruco/dataset/ -m ~/datasets/red_book_aruco/dataset/1528885039597.obj -i ~/datasets/red_book_aruco/dataset/calibrations/camera.yaml -ms 0.082 -si 15
```

### Projection based color balancing

```bash
clear && test/projection_based_color_balancing_oc_dataset.py -p ~/datasets/red_book_aruco/dataset/ -m ~/datasets/red_book_aruco/dataset/1528885039597.obj -i ~/datasets/red_book_aruco/dataset/calibrations/camera.yaml -ms 0.082 -si 25 -sv 50 -z 0.1 -vo
```
