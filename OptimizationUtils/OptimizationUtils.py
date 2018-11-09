"""
This is a long, multiline description
"""

import argparse  # to read command line arguments
#to list images in a folder
import random
#import glob
#import json
#import shutil
#import yaml
#from collections import namedtuple
#from copy import deepcopy
#from matplotlib import cm
#import openmesh as om
#from tqdm import tqdm
#from model_viewer import *
#from rgbd_camera import *
#from itertools import combinations
from scipy.spatial import distance  #Compute euclidean distance
from scipy.sparse import lil_matrix 
from scipy.optimize import least_squares

# ------------------------
##   DATA STRUCTURES   ##
# ------------------------
from collections import namedtuple
ParamT = namedtuple('ParamT', 'idx data_key getter setter')

# ------------------------
## FUNCTION DEFINITION ##
# ------------------------

# ------------------------
###   BASE CLASSES    ###
# ------------------------

# ------------------------
### CLASS DEFINITION  ###
# ------------------------

class Optimizer():
    """ Class to color correct
    """

    def __init__(self):
        """Implements a color corrector
           Args: 
           Returns:
              None for now 
        """
        self.static_data = {}
        self.params = {}    
        self.x = []

    def pushScalarParam(self, name, data_key, getter, setter):

        if name in self.params:
            raise ValueError('Scalar param ' + name + ' already exits.')

        self.params[name] = ParamT(len(self.x), data_key, getter, setter)
        self.x.append(getter(self.static_data[data_key]))

        print('Pushed scalar param ' + name)

    def fromModelToX(self):
        for name in self.params:
            idx, data_key, getter, _ = self.params[name]
            value = getter(self.static_data[data_key])
            self.x[idx] = value

    def fromXToModel(self):
        for name in self.params:
            print("from x to model param " + str(name))
            idx, data_key, _, setter = self.params[name]
            value = self.x[idx]
            print('value = ' + str(value))
            print('setter = ' + str(setter))
            setter(self.static_data[data_key], value)         

    def addStaticData(self, name, data):
        """ Should be a dictionary containing every static data to be used by the cost function"""
        print('Adding new data "' + name + '" to static data')
        self.static_data[name] = data

    def setObjectiveFunction(self, function_handle):
        self.objective_function = function_handle

    def callObjectiveFunction(self):
        self.objective_function(self.static_data)

    def internalObjectiveFunction(self, x):
        self.x = x
        self.fromXToModel()
        error = self.objective_function(self.static_data)
        return error

    def printXandModel(self, opt):
        print('-----')
        print('x0 = ' + str(self.x))
        for i, camera in enumerate(self.static_data['dataset'].cameras):
            print('camera ' + str(i) + ' = ' + str(camera.rgb.bias))
        print('-----')

    def startOptimization(self):
        # res = least_squares(costFunction, x0, verbose=2, jac_sparsity=A, x_scale='jac', ftol=1e-4, xtol=1e-4, bounds=bounds, method='trf', args=(X, Pc, detections, args, handles, handle_fun, s))

        res = least_squares(self.internalObjectiveFunction, self.x, verbose=2, x_scale='jac', ftol=1e-6, xtol=1e-6, gtol=1e-6, diff_step=1e0,  method='trf', args=())



