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
import numpy as np
import random
import KeyPressManager.KeyPressManager

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
        self.groups = OrderedDict()  # groups of params an ordered dict where key={name} and value = namedtuple('ParamT')
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
            raise ValueError('For scalar parameters, getter must return a list of lenght 1. Returned list = ' + str(
                value) + ' of type ' + str(type(value)))

        param_names = [group_name]  # a single parameter with the same name as the group
        idx = [len(self.x)]
        self.groups[group_name] = ParamT(param_names, idx, data_key, getter, setter, [bound_max],
                                         [bound_min])  # add to group dict
        self.x.append(value[0])  # set initial value in x using the value from the data model
        print('Pushed scalar param ' + group_name + ' to group ' + group_name)

    def pushParamVector3(self, group_name, data_key, getter, setter, bound_max=(+inf, +inf, +inf),
                         bound_min=(-inf, -inf, -inf), sufix=['x', 'y', 'z']):
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

        if not data_key in self.data:  # Check if we have the data_key in the data dictionary
            raise ValueError('Dataset ' + data_key + ' does not exist. Cannot add group ' + group_name + '.')

        if not len(bound_max) == 3:  # check size of bound_max
            raise ValueError('bound_max ' + str(bound_max) + ' must be a tuple of size 3, e.g. (max_x, max_y, max_z).')

        if not len(bound_min) == 3:  # check size of bound_min
            raise ValueError('bound_min ' + str(bound_min) + ' must be a tuple of size 3, e.g. (min_x, min_y, min_z).')

        if not len(sufix) == 3:
            raise ValueError('sufix ' + str(sufix) + ' must be a list of size 3, e.g. ["x", "y", "z"].')

        idxs = range(len(self.x), len(self.x) + 3)  # Compute value of indices

        param_names = [group_name + sufix[0], group_name + sufix[1], group_name + sufix[2]]

        self.groups[group_name] = ParamT(param_names, idxs, data_key, getter, setter, bound_max,
                                         bound_min)  # add to params dict
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
        self.internalObjectiveFunction(self.x)

    def internalObjectiveFunction(self, x):
        """
        A wrapper around the custom given objective function which maps the x vector to the model before calling the
        objetive function and after the call
        :param x: the parameters vector
        """
        self.x = x
        self.fromXToData()
        error = self.objective_function(self.data)

        if self.counter >= self.visualization_function_iterations:
            self.visualization_function(self.data)
            self.counter = 0

            print('AvgError = ' + str(np.average(error)))

        self.counter += 1

        return error

    def startOptimization(self, optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-8,
                                                      'diff_step': 1e-3}):
        self.setFirstGuess()

        bounds_min = []
        bounds_max = []
        for name in self.groups:
            _, _, _, _, _, max, min = self.groups[name]
            bounds_max.extend(max)
            bounds_min.extend(min)

        self.result = least_squares(self.internalObjectiveFunction, self.x, verbose=2, jac_sparsity=self.sparse_matrix,
                                    bounds=(bounds_min, bounds_max), method='trf', args=(), **optimization_options)
        self.xf = deepcopy(list(self.result.x))
        self.finalOptimizationReport()

        # wm = KeyPressManager.KeyPressManager.WindowManager()
        # if wm.waitForKey(self.fig):
        #     exit(0)

    # def setFigure(self, figure):
    #     self.figure = figure

    def finalOptimizationReport(self):
        """Just print some info and show the images"""
        print('\n-------------\nOptimization finished')
        print(self.result)
        # self.printX(preamble_text='\nInitial value of parameters', x=self.x0)
        # self.printX(preamble_text='\nFinal value of parameters', x=self.xf)

        self.fromXToData(self.xf)
        self.visualization_function(self.data)
        # cv2.waitKey(20)

    # ---------------------------
    # Utilities
    # ---------------------------
    def addNoiseToX(self, noise=0.1, x=None):
        if x is None:
            x = self.x

        return x * np.array([random.uniform(1 - noise, 1 + noise) for _ in xrange(len(x))], dtype=np.float)

    def getParameters(self):
        params = []
        for group_name, group in self.groups.items():
            params.extend(group.param_names)
        return params

    def getParamsContainingPattern(self, pattern):
        params = []
        for group_name, group in self.groups.items():
            for i, param_name in enumerate(group.param_names):
                if pattern in param_name:
                    params.append(param_name)
        return params

    def setFirstGuess(self):
        self.x0 = deepcopy(self.x)

    def fromDataToX(self, x=None):
        """Copies values of all parameters from the data to the vector x"""
        if x is None:
            x = self.x

        for group_name, group in self.groups.items():
            values = group.getter(self.data[group.data_key])
            for i, idx in enumerate(group.idx):
                x[idx] = values[i]

    def fromXToData(self, x=None):
        """Copies values of all parameters from vector x to the data"""
        if x is None:
            x = self.x

        for group_name, group in self.groups.items():
            values = []
            for idx in group.idx:
                values.append(x[idx])

            group.setter(self.data[group.data_key], values)

    def computeSparseMatrix(self):

        params = self.getParameters()
        self.sparse_matrix = lil_matrix((len(self.residuals), len(params)), dtype=int)

        for i, key in enumerate(self.residuals):
            for param in self.residuals[key]:
                print("param = " + param)
                for group_name, group in self.groups.items():
                    if param in group.param_names:
                        idx_in_group = group.param_names.index(param)
                        print("param_names = " + str(group.param_names))
                        idx = group.idx[idx_in_group]
                        print("group.idx = " + str(group.idx))
                        print("idx_in_group = " + str(idx_in_group))
                        print("idx = " + str(idx))
                        self.sparse_matrix[i, idx] = 1

        print('Sparsity matrix:')
        data_frame = pandas.DataFrame(self.sparse_matrix.toarray(), self.residuals, params)
        print(data_frame)
        data_frame.to_csv('sparse_matrix.csv')

    # ---------------------------
    # Print and display
    # ---------------------------
    def printX(self, x=None):
        if x is None:
            x = self.x

        for group_name, group in self.groups.items():
            print('Group ' + str(group_name) + ' has parameters:')
            values_in_data = group.getter(self.data[group.data_key])
            for i, param_name in enumerate(group.param_names):
                print('--- ' + str(param_name) + ' = ' + str(values_in_data[i]) + ' (in data) ' + str(
                    x[group.idx[i]]) + ' (in x)')

        print(self.x)

    def printParameters(self, x=None):
        if x is None:
            x = self.x

        # Build a panda data frame and then print a nice table
        rows = []  # get a list of parameters
        table = []
        for group_name, group in self.groups.items():
            values_in_data = group.getter(self.data[group.data_key])
            for i, param_name in enumerate(group.param_names):
                rows.append(param_name)
                table.append([group_name, x[group.idx[i]], values_in_data[i]])

        print('Parameters:')
        print(pandas.DataFrame(table, rows, ['Group', 'x', 'data']))

    def printModel(self):
        print('There are ' + str(len(self.data)) + ' data models stored: ' + str(self.data))

    def printXAndModel(self):
        self.printX()
        self.printModel()
