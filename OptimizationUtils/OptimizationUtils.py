"""
This is a long, multiline description
"""

import argparse  # to read command line arguments
#to list images in a folder
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
ParamT = namedtuple('ParamT', 'idx binding')

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
        self.x0 = []

    def pushScalarParam(self, name, binding):
        self.params[name] = ParamT(len(self.x0), binding)
        self.x0.append(self.params[name].binding)

    #def fromX(self):
        #for key, value in self.params.iteritems():
            

    def addStaticData(self, name, data):
        """ Should be a dictionary containing every static data to be used by the cost function"""
        print('Adding new data "' + name + '" to static data')
        self.static_data[name] = data

    def setObjectiveFunction(self, function_handle):
        self.objective_function = function_handle

    def callObjectiveFunction(self, x):
        self.objective_function(x, self.static_data)





