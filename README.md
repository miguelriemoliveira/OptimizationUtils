# OptimizationUtils
A set of utilities for using the python scipy optimizer functions

# Usage

There are several examples. Here is how to launch them:

### test_ocdataset_color_correction

Uses the OCDatasetLoader to load an OC dataset and runs a color balancing optimization using images from the first three cameras.

```bash
test/test_ocdataset_color_correction.py -p ~/datasets/red_book2/dataset/ -m ~/datasets/red_book2/dataset/1528188687058_simplified_decimated.obj -i ~/datasets/red_book2/dataset/calibrations/camera.yaml -si 5
```
