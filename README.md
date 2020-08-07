# OptimizationUtils
A set of utilities for quickly and efficiently setup complex optimization problems. 


# How to setup an optimization problem

The goal of OptimizationUtils is to facilitate the configuration of an optimization problem. The system works by declaring an optimizer class and them proceeding to configure the optimizer before starting up the optimization. To instantiate an optimizer:

```python 
import OptimizationUtils.OptimizationUtils as OptimizationUtils
opt = OptimizationUtils.Optimizer()
```

##### Set data models

One of the biggest troubles is the need to put the parameters to be optimized in a list. Often these parameters are very different, and putting them altogether in a list while having to keep track of the indices of each is a cumbersome and uninteresting task.

OptimizationUtils solves this by allowing you to **use your own data structures as optimization parameters**. This is achieved by maintaining an internal mapping between some of the variables in your data structures (to which we refer as **data models**) and a list based representation of the parameters which is given to the optimizer. 

Suppose you have an instance of a class, containing two variables that you want to optimize:

```python 
Class Dog:  # declare a class dog
    __init__(weight, height):
        self.weight = weight
        self.height = height

dog = Dog(weigth=5.4, height=0.2)  # instantiate a large dog
```

and that you have two other variables that are also to be optimized, but this time are contained in a dictionary:

```python 
cat = {'weight': 3.2, 'height': 0.1} # define a tiny cat using a dictionary
```

to use these variables you have to provide both data models to the optimizer:

```python 
opt.addDataModel('dog', dog)
opt.addDataModel('cat', cat)
```

##### Define parameters to be optimized

Then, we define each of the parameters to be optimized. To do so one must define how the parameter is accessed and written from / to the data model. This is done by defining **getters** and **setters**:

```python 
def getDogWeightOrHeight(data, property):
    if property is 'weight':
        return data.weight
    elif property is 'height':
        return data.height

def setDogWeightOrHeight(data, value, property):
    if property is 'weight':
        data.weight = value
    elif property is 'height':
        data.height = value
```

now the parameters dog weight and dog height can be defined:

```python 
from functools import partial
opt.pushParamScalar(group_name='dog_weight', data_key='dog',
                    getter=partial(getDogWeightOrHeight, property='weight'), 
                    setter=partial(setDogWeightOrHeight, property='weight'))

opt.pushParamScalar(group_name='dog_height', data_key='dog',
                    getter=partial(getDogWeightOrHeight, property='height'), 
                    setter=partial(setDogWeightOrHeight, property='height'))
```

It is also possible to define groups of parameters, which are parameters that share the same getter and setter. One typical example of a group of parameters is a pose, which contains variables for the translation and rotation components. Lets define a group of parameters for the cat: 

```python 
def getCatWeightAndHeight(data):
    return [data['weight'], data['height']]

def setCatWeightAndHeight(data, values):
    data['weight'] = values[0]
    data['height'] = values[1]

opt.pushParamGroup(group_name='cat', data_key='cat',
                    getter=getCatWeightAndHeight, 
                    setter=setCatWeightAndHeight,
                    suffix=['_weight', '_height'])
```

##### Define the objective function

Now you write the objective function using your own data models, rather than some confusing linear array with thousands of parameters.

This is possible because OptimizationUtils updates the values contained in your own data models by copying from the parameter vector being optimized. This greatly facilitates the writing of the objective function, provided you are any good at defining easy to use data structures, that's on you. 

Suppose that you have zero clue about the biometric of cats and dogs and aim to have a dog and a cat that weight the same, and stand at the same height.  I know, I known, those sound a bit eccentric or even ridiculous, but hey, those are your whims, not ours, so don't complain. Having this goal in mind you could write the following objective function:

```python 
def objectiveFunction(data_models):
    dog = data_models['dog']
    cat = data_models['cat']
    residuals = {} 

    residuals['weight_diference'] = dog.weight - cat['weight']
    residuals['height_diference'] = dog.height - cat['height']
    return residuals

opt.setObjectiveFunction(objectiveFunction)
```

Notice we use the argument data_models to extract the updated variables in our own data format. Then, two residuals are created in a dictionary and that dictionary is returned.

##### Defining the residuals

We must also define the residuals that are output by the objective function. For each residual we must identify which parameters  influence that residual (for sparse optimization problems):

```python 
params = opt.getParamsContainingPattern('weight') # get all weight related parameters
opt.pushResidual(name='weight_diference', params=params) 

params = opt.getParamsContainingPattern('height') # get all height related parameters
opt.pushResidual(name='height_diference', params=params) 
```
 
 ##### Computing the sparse matrix
 
 For sparse optimization problems, i.e. those in which not all parameters affect all residuals, a sparse matrix is used to map which parameters affect which residuals. Having such information considerably speeds up the optimization: there is no need to estimate the gradient for nonexistent parameter - residual pairs.
 
 Having defined the parameters and residuals, the sparse matrix is computed automatically. Notice that for large and complex optimization problems computing this matrix is not straightforward:

```python 
opt.computeSparseMatrix()
```

which, for our dog - cat problem would return this:

```bash
            |              residuals               | 
parameters  |  weight_diference | height_diference | 
----------------------------------------------------
dog_weight  |         1         |        0         |
dog_height  |         0         |        1         |
cat_weight  |         1         |        0         |
cat_height  |         0         |        1         |
----------------------------------------------------
```

##### Visualizing the optimization

One important aspect of monitoring an optimization procedure is the ability to visualize the procedure in real time. OptimizationUtils provides two general purpose visualizations which display the evolution of the residuals over time, as well as the evolution of total error over time. These are constructed using the information about parameters and residuals entered before.

Total Error vs Iterations | Residuals vs Iterations
------------- | -------------
<img align="center" src="https://github.com/miguelriemoliveira/OptimizationUtils/blob/master/docs/total_error.png" width="450"/>  | <img align="center" src="https://github.com/miguelriemoliveira/OptimizationUtils/blob/master/docs/optimization_residuals.png" width="450"/>

Besides these embedded general visualizations, you can design your own visualizations. To do this, create a function that produces the visualization you'd like. This function is called every n times the objective function is called. 


##### Starting the optimization

To run the optimization use:

```python 
opt.startOptimization(optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 
                        'xtol': 1e-6, 'gtol':1e-6, 'diff_step': None})
```

The optimization is a least squares optimization implemented in [scypy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html). The possible options are listen in the function's page.

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

# Examples

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
