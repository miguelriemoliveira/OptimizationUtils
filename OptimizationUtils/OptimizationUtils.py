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
ParamT = namedtuple('ParamT', 'idx data_key getter setter bound_max bound_min')


# -------------------------------------------------------------------------------
# CLASS
# -------------------------------------------------------------------------------
class Optimizer:

    def __init__(self):
        self.counter = 0
        self.data = {}  # a dict with a set of variables or structures to be used by the objective function
        self.groups = {}
        self.params = OrderedDict()  # an ordered dict where key={name} and value = namedtuple('ParamT')
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
        if name in self.params:  # Cannot add a parameter that already exists
            raise ValueError('Data ' + name + ' already exits in model dict.')
        else:
            self.data[name] = data
            print('Added data ' + name + ' to model dict.')

    def pushScalarParam(self, name, data_key, getter, setter, bound_max=+inf, bound_min=-inf):
        """
        Pushes a new parameter to the parameter vector
        :param name: the name of the parameter
        :param data_key: the key of the model into which the parameter maps
        :param getter: a function to retrieve the parameter value from the model
        :param setter: a function to set the parameter value from the model
        :param bound_max: max value the parameter may take
        :param bound_min: min value the parameter may take
        """
        if name in self.params:  # Cannot add a parameter that already exists
            raise ValueError('Scalar param ' + name + ' already exists. Cannot add it.')

        if not data_key in self.data:
            raise ValueError('Dataset ' + data_key + ' does not exist. Cannot add param ' + name + '.')

        self.params[name] = ParamT(len(self.x), data_key, getter, setter, bound_max, bound_min)  # add to params dict
        self.x.append(getter(self.data[data_key]))  # set initial value in x
        print('Pushed scalar param ' + name)

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
        for name in self.params:
            _, _, _, _, max, min = self.params[name]
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
        for name in self.params:
            idx, data_key, getter, _ = self.params[name]
            value = getter(self.data[data_key])
            x[idx] = value

    def fromXToModel(self, x=None):
        """Copies values of all parameters from vector x to model"""
        if x is None:
            x = self.x
        for name in self.params:
            idx, data_key, _, setter, _, _ = self.params[name]
            value = x[idx]
            setter(self.data[data_key], value)

    def computeSparseMatrix(self):
        self.sparse_matrix = lil_matrix((len(self.residuals), len(self.params)), dtype=int)

        for i, key in enumerate(self.residuals):
            for param in self.residuals[key]:
                idx, _, _, _, _, _ = self.params[param]
                self.sparse_matrix[i, idx] = 1

        print('Sparsity matrix:')
        print(pandas.DataFrame(self.sparse_matrix.toarray(), self.residuals, self.params.keys()))

    # ---------------------------
    # Print and display
    # ---------------------------
    def printX(self, text='Parameter vector:', x=None):
        if x is None:
            x = self.x

        print(text)
        for i, param_value in enumerate(x):
            for name in self.params:
                idx, _, _, _, _, _ = self.params[name]
                if idx == i:
                    print('   ' + name + ' = ' + str(param_value))

    def printModel(self):
        print('Model =\n' + str(self.data))

    def printXAndModel(self):
        self.printX()
        self.printModel()
