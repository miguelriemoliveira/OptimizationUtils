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
test/color_balancing_oc_dataset.py -p ~/datasets/red_book2/ -m ~/datasets/red_book2/1528188687058_simplified_decimated.obj -i ~/datasets/red_book2/calibrations/camera.yaml -si 5
```

### Camera pose optimization using an OC dataset

Uses the OCDatasetLoader to load an OC dataset and runs a camera pose optimization.

```bash
test/camera_pose_oc_dataset.py -p ~/datasets/red_book_aruco/ -m ~/datasets/red_book_aruco/1528885039597.obj -i ~/datasets/red_book_aruco/calibrations/camera.yaml -ms 0.082 -si 15
```

to view the aruco detections run:

```bash
test/camera_pose_oc_dataset.py -p ~/datasets/red_book_aruco/ -m ~/datasets/red_book_aruco/1528885039597.obj -i ~/datasets/red_book_aruco/calibrations/camera.yaml -ms 0.082 -vad -va3d -si 15
```

and to skip images or select only a few arucos

```bash
test/camera_pose_oc_dataset.py -p ~/datasets/lobby2/ -m ~/datasets/lobby2/1553614275334.obj -i ~/datasets/lobby2/calibrations/camera.yaml -ms 0.082 -si 1 -vo -csf 'lambda name: int(name)<20' -mnai 1 -asf 'lambda id: int(id) > 560'
```

### Pose and color optimization using an OC dataset

Uses the OCDatasetLoader to load an OC dataset and runs a camera pose plus camera color optimization.

```bash
test/pose_and_color_oc_dataset.py -p ~/datasets/red_book_aruco/ -m ~/datasets/red_book_aruco/1528885039597.obj -i ~/datasets/red_book_aruco/calibrations/camera.yaml -ms 0.082 -si 15
```

### Projection based color balancing

```bash
clear && test/projection_based_color_balancing_oc_dataset.py -p ~/datasets/red_book_aruco/ -m ~/datasets/red_book_aruco/1528885039597.obj -i ~/datasets/red_book_aruco/calibrations/camera.yaml -ms 0.082 -si 25 -sv 50 -z 0.1 -vo
```
 
### to read json file in your datasets 
```
  test/sensor_pose_json.py -json <json_:path_to_your_json>
```

### Calibration of sensors in the atlascar

To generate a dataset

```bash
roslaunch interactive_calibration atlascar2_calibration.launch
```

and then:

```bash
clear && rosrun interactive_calibration collect_and_label_data.py -w car_center -o /home/mike/datasets/calibration_test2 -s .5
```

You can visualize the json file by copying to 

https://jsoneditoronline.org/#/

and copy the contents of the /home/mike/datasets/calibration_test2/data_collected.json to the left window.


```bash
test/sensor_pose_json_v2/main.py -json ~/datasets/calib_without_fg/calibration_complete_nofg/data_collected.json -cradius .5 -csize 0.1054 -cnumx 8 -cnumy 6 -vo
```

If you want to filter out some sensors or collections you may use the sensor selection function (ssf) or collection selection function (csf) as follows:

```bash
test/sensor_pose_json_v2/main.py -json ~/datasets/calib_without_fg/calibration_complete_nofg/data_collected.json -cradius .5 -csize 0.1054 -cnumx 8 -cnumy 6 -vo -ssf "lambda name: name in ['left_laser', 'right_laser']" -csf "lambda name: int(name) < 1"
```
