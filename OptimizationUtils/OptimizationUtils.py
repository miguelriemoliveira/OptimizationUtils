# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
from collections import namedtuple, OrderedDict
from copy import deepcopy
import pandas
import cv2
from numpy import inf
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# ------------------------
# DATA STRUCTURES   ##
# ------------------------
ParamT = namedtuple('ParamT', 'param_names idx data_key getter setter bound_max bound_min')


# -------------------------------------------------------------------------------
# CLASS
# -------------------------------------------------------------------------------
class Optimizer:

    def __init__(self):
        self.counter = 0
        self.data = {}  # a dict with a set of variables or structures to be used by the objective function
        self.groups = OrderedDict()  #groups of params an ordered dict where key={name} and value = namedtuple('ParamT')
        self.x = []  # a list of floats (the actual parameters)
        self.x0 = []  # the initial value of the parameters
        self.xf = []  # the final value of the parameters
        self.residuals = OrderedDict()  # ordered dict: key={residual} value = [params that influence this residual]
        self.sparse_matrix = None
        self.result = None  # to contain the optimization result
        self.objective_function = None  # to contain the objective function
        self.visualization_function = None  # to contain the visualization function
        self.visualization_function_iterations = 0

    # ---------------------------
    # Optimizer configuration
    # ---------------------------
    def addModelData(self, name, data):
        """ Should be a dictionary containing every static data to be used by the cost function"""
        if name in self.data:  # Cannot add a parameter that already exists
            raise ValueError('Data ' + name + ' already exits in model dict.')
        else:
            self.data[name] = data
            print('Added data ' + name + ' to model dict.')

    def pushParamScalar(self, group_name, data_key, getter, setter, bound_max=+inf, bound_min=-inf):
        """
        Pushes a new scalar parameter to the parameter vector. The parameter group contains a single element.
        Group name is the same as parameter name.
        :param group_name: the name of the parameter
        :param data_key: the key of the model into which the parameter maps
        :param getter: a function to retrieve the parameter value from the model
        :param setter: a function to set the parameter value from the model
        :param bound_max: max value the parameter may take
        :param bound_min: min value the parameter may take
        """
        if group_name in self.groups:  # Cannot add a parameter that already exists
            raise ValueError('Scalar param ' + group_name + ' already exists. Cannot add it.')

        if not data_key in self.data:
            raise ValueError('Dataset ' + data_key + ' does not exist. Cannot add group ' + group_name + '.')

        value = getter(self.data[data_key])
        if not type(value) is list or not (len(value) == 1):
            raise ValueError('For scalar parameters, getter must return a list of lenght 1. Returned list = ' + str(value) + ' of type ' + str(type(value)))

        param_names = [group_name] # a single parameter with the same name as the group
        idx = len(self.x)
        self.groups[group_name] = ParamT(param_names, idx, data_key, getter, setter, [bound_max], [bound_min])  # add to group dict
        self.x.append(value)  # set initial value in x using the value from the data model
        print('Pushed scalar param ' + group_name + ' to group ' + group_name)

    def pushParamTranslation(self, group_name, data_key, getter, setter, bound_max=(+inf, +inf, +inf), bound_min=(-inf, -inf, -inf)):
        """
        Pushes a new parameter group of type translation to the parameter vector.
        There will be 3 parameters, *_tx, *_ty, *_tz per translation group
        :param group_name: the name of the group of parameters, which will have their name derived from the group name.
        :param data_key: the key of the model into which the parameters map
        :param getter: a function to retrieve the parameter value from the model
        :param setter: a function to set the parameter value from the model
        :param bound_max: a tuple (max_x, max_y, max_z)
        :param bound_min: a tuple (min_x, min_y, min_z)
        """
        if group_name in self.groups:  # Cannot add a parameter that already exists
            raise ValueError('Group ' + group_name + ' already exists. Cannot add it.')

        if not data_key in self.data: # Check if we have the data_key in the data dictionary
            raise ValueError('Dataset ' + data_key + ' does not exist. Cannot add group ' + group_name + '.')

        if not len(bound_max) == 3: # check size of bound_max
            raise ValueError('bound_max ' + str(bound_max) + ' must be a tuple of size 3, e.g. (max_x, max_y, max_z).')

        if not len(bound_min) == 3: # check size of bound_min
            raise ValueError('bound_min ' + str(bound_min) + ' must be a tuple of size 3, e.g. (min_x, min_y, min_z).')

        idxs = range(len(self.x), len(self.x)+3) # Compute value of indices

        param_names = [group_name + '_tx', group_name + '_ty', group_name + '_tz']

        self.groups[group_name] = ParamT(param_names, idxs,  data_key, getter, setter, bound_max, bound_min)  # add to params dict
        values = getter(self.data[data_key])
        for value in values:
            self.x.append(value)  # set initial value in x
        print('Pushed translation group ' + group_name + ' with params ' + str(param_names))

    def pushResidual(self, name, params=None):
        """
        Adds a new residual to the existing list of residuals
        :param name: name of residual
        :param params: list of parameter names which affect this residual
        """
        # TODO check if all listed params exist in the self.params
        self.residuals[name] = params

    def setObjectiveFunction(self, function_handle):
        self.objective_function = function_handle

    def setVisualizationFunction(self, function_handle, n_iterations=0):
        self.visualization_function = function_handle
        self.visualization_function_iterations = n_iterations

    # ---------------------------
    # Optimization functions
    # ---------------------------
    def callObjectiveFunction(self):
        self.objective_function(self.data)

    def internalObjectiveFunction(self, x):
        """
        A wrapper around the custom given objective function which maps the x vector to the model before calling the
        objetive function and after the call
        :param x: the parameters vector
        """
        self.x = x
        self.fromXToModel()
        error = self.objective_function(self.data)

        if self.counter >= self.visualization_function_iterations:
            self.visualization_function(self.data)
            self.counter = 0
        self.counter += self.counter

        return error

    def startOptimization(self, optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-8,
                                                      'diff_step': 1e0}):
        self.setFirstGuess()
        bounds_min = []
        bounds_max = []
        for name in self.groups:
            _, _, _, _, max, min = self.groups[name]
            bounds_max.append(max)
            bounds_min.append(min)

        self.result = least_squares(self.internalObjectiveFunction, self.x, verbose=2, jac_sparsity=self.sparse_matrix,
                                    bounds=(bounds_min, bounds_max), method='trf', args=(), **optimization_options)
        self.xf = list(self.result.x)
        self.finalOptimizationReport()

    def finalOptimizationReport(self):
        """Just print some info and show the images"""
        print('\n-------------\nOptimization finished')
        print(self.result)
        self.printX(text='\nInitial value of parameters', x=self.x0)
        self.printX(text='\nFinal value of parameters', x=self.xf)

        self.fromXToModel(self.xf)
        self.visualization_function(self.data)
        cv2.waitKey(0)

    # ---------------------------
    # Utilities
    # ---------------------------
    def setFirstGuess(self):
        self.x0 = deepcopy(self.x)

    def fromModelToX(self, x=None):
        """Copies values of all parameters from model to vector x"""
        if x is None:
            x = self.x
        for name in self.groups:
            idx, data_key, getter, _ = self.groups[name]
            value = getter(self.data[data_key])
            x[idx] = value

    def fromXToModel(self, x=None):
        """Copies values of all parameters from vector x to model"""
        if x is None:
            x = self.x
        for name in self.groups:
            idx, data_key, _, setter, _, _ = self.groups[name]
            value = x[idx]
            setter(self.data[data_key], value)

    def computeSparseMatrix(self):
        self.sparse_matrix = lil_matrix((len(self.residuals), len(self.groups)), dtype=int)

        for i, key in enumerate(self.residuals):
            for param in self.residuals[key]:
                idx, _, _, _, _, _ = self.groups[param]
                self.sparse_matrix[i, idx] = 1

        print('Sparsity matrix:')
        print(pandas.DataFrame(self.sparse_matrix.toarray(), self.residuals, self.groups.keys()))

    # ---------------------------
    # Print and display
    # ---------------------------
    def printX(self, text='Parameter vector:', x=None):
        if x is None:
            x = self.x

        print(text)
        for group_name, group in self.groups.items():
            print('Group ' + str(group_name) + ':')
            values = group.getter(self.data[group.data_key])
            print('values = ' + str(values))
            print('Group params:' + str(group.param_names))
            for i, param_name in enumerate(group.param_names):
                print('   Param ' + str(param_name) + ' = ' + str(values[i]))

    def printModel(self):
        print('There are ' + str(len(self.data)) + ' data models stored: ' + str(self.data))

    def printXAndModel(self):
        self.printX()
        self.printModel()
