#!/usr/bin/env python
"""
This example shows an optimizer working with a set of n cameras, changing their pose so that the reprojection error is
minimized.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import OCDatasetLoader.OCArucoDetector as OCArucoDetector
import OptimizationUtils.OptimizationUtils as OptimizationUtils
import KeyPressManager.KeyPressManager as KeyPressManager
import OptimizationUtils.utilities as utilities
import numpy as np
import matplotlib.pyplot as plt
import plyfile as plyfile
import cv2
import argparse
import subprocess
import os
import shutil
from copy import deepcopy
from functools import partial
from matplotlib import cm
from scipy.spatial.distance import euclidean
from open3d import *
