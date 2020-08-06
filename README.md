# OptimizationUtils
A set of utilities for using the python scipy optimizer functions

# Installation

You can install from source
```bash
git clone https://github.com/miguelriemoliveira/OptimizationUtils.git
cd OptimizationUtils
python setup.py install --user
```

You can also use `pip` to install from source
```bash
git clone https://github.com/miguelriemoliveira/OptimizationUtils.git
pip install OptimizationUtils
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
roslaunch atom_calibration atlascar2_calibration.launch read_first_guess:=true
```

and then:

```bash
rosrun atom_calibration collect_data.py -o ~/datasets/calib_complete_fg_v2 -s .5 -c ~/catkin_ws/src/AtlasCarCalibration/atom_calibration/calibrations/atlascar2/atlascar2_calibration.json
```

You can visualize the json file by copying to 

https://jsoneditoronline.org/#/

and copy the contents of the ~/datasets/calib_complete_fg_v2/data_collected.json to the left window.


```bash
test/sensor_pose_json_v2/main.py -json ~/datasets/calibration_test2/data_collected.json -vo
```

If you want to filter out some sensors or collections you may use the sensor selection function (ssf) or collection selection function (csf) as follows:

```bash
test/sensor_pose_json_v2/main.py -json ~/datasets/calib_complete_fg_v2/data_collected.json -ssf "lambda name: name in ['top_left_camera', 'top_right_camera']"
```


### Calibration of sensors in the atlascar (with RVIZ visualization)


First launch rviz. There a dedicated launch file for this.

```bash
roslaunch atom_calibration atlascar2_view_optimization.launch 
```

```bash
test/sensor_pose_json_v2/main.py -json ~/datasets/calibration_test2/data_collected.json -vo -si
```


### Calibration results visualization

Comparing this optimization procedure with some openCV tools:

Calibrating using openCV stereo calibration (https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=stereo#cv2.stereoCalibrate):
```bash
test/sensor_pose_json_v2/stereocalib_v2.py -json ~/datasets/calib_complete_fg_v2/data_collected.json -cradius .5 -csize 0.101 -cnumx 9 -cnumy 6 -fs top_left_camera -ss top_right_camera

```
Calibrating using openCV calibrate camera (https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera):
```bash
test/sensor_pose_json_v2/calibcamera.py -json ~/datasets/calib_complete_fg_v2/data_collected.json -cradius .5 -csize 0.101 -cnumx 9 -cnumy 6 -fs top_left_camera -ss top_right_camera 
```

Transforming the kabir2 calibration txt file in a equal json format than the previous mentioned procedures (the original json and the kalibr txt files are required):
```bash
test/sensor_pose_json_v2/kalibr2_txt_to_json.py -json ~/datasets/dataset_23_dez_2019/original.json -kalibr ~/datasets/dataset_23_dez_2019/results-cam-for_kalibr2.txt -cnumx 9 -cnumy 6 -csize 0.101 
```


In order to see the difference between the image points and the reprojected points (for each collection, for each procedure) you must run the following:

```bash
test/sensor_pose_json_v2/results_visualization.py -json_opt_left test/sensor_pose_json_v2/results/dataset_sensors_results_top_left_camera.json -json_opt_right test/sensor_pose_json_v2/results/dataset_sensors_results_top_right_camera.json -json_stereo test/sensor_pose_json_v2/results/opencv_stereocalib.json -json_calibcam test/sensor_pose_json_v2/results/opencv_calibcamera.json -json_kalibr test/sensor_pose_json_v2/results/kalibr2_calib.json -fs top_left_camera -ss top_right_camera

```
You should give the final json of each one of the distinct calibration procedures. 
Beside this, you must choose wich one is the first sensor (fs) and the second sensor (ss). 
The points will be projected from the first sensor image (pixs) to the second sensor image (pixs), where the difference between the points will be measured.
