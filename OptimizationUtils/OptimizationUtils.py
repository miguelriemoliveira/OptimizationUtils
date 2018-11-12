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
##   DATA STRUCTURES   ##
# ------------------------
ParamT = namedtuple('ParamT', 'idx data_key getter setter')


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------
class Optimizer:

    def __init__(self):
        self.counter = 0
        self.model = {}  # a dict with a set of variables or structures to be used by the objective function
        self.params = OrderedDict()  # an ordered dict where key={name} and value = namedtuple('ParamT')
        self.x = []  # a list of floats (the actual parameters)
        self.x0 = []  # the initial value of the parameters
        self.bound_min = []  # min values for x
        self.bound_max = []  # max values for x
        self.residuals = []
        self.residual_params = {}

    def setFirstGuess(self):
        self.x0 = deepcopy(self.x)

    def pushScalarParam(self, name, model_key, getter, setter, bound_max=+inf, bound_min=-inf):
        """
        Pushes a new parameter to the parameter vector
        :param name: the name of the parameter
        :param model_key: the key of the model into which the parameter maps
        :param getter: a function to retrieve the parameter value from the model
        :param setter: a function to set the parameter value from the model
        :param bound_max: max value the parameter may take
        :param bound_min: min value the parameter may take
        """
        if name in self.params:  # Cannot add a parameter that already exists
            raise ValueError('Scalar param ' + name + ' already exists.')

        self.params[name] = ParamT(len(self.x), model_key, getter, setter)  # add to params dict
        self.x.append(getter(self.model[model_key]))  # set initial value in x
        self.bound_min.append(bound_min)  # set min value for this param
        self.bound_max.append(bound_max)  # set max value for this param

        print('Pushed scalar param ' + name)

    def pushResidual(self, name, params=None):
        """
        Adds a new residual to the existing list of residuals
        :param name: name of residual
        :param params: list of parameter names which affect this residual
        """
        self.residuals.append(name)
        self.residual_params[name] = params

    def fromModelToX(self):
        """Copies values of all parameters from model to vector x"""
        for name in self.params:
            idx, data_key, getter, _ = self.params[name]
            value = getter(self.model[data_key])
            self.x[idx] = value

    def fromXToModel(self, x=None):
        """Copies values of all parameters from vector x to model"""
        if x is None:
            x = self.x

        for name in self.params:
            idx, data_key, _, setter = self.params[name]
            value = x[idx]
            setter(self.model[data_key], value)

    def addStaticData(self, name, data):
        """ Should be a dictionary containing every static data to be used by the cost function"""
        if name in self.params:  # Cannot add a parameter that already exists
            raise ValueError(name + ' already exits in model dict.')
        else:
            self.model[name] = data
            print('Added ' + name + ' to model dict.')

    # noinspection PyAttributeOutsideInit
    def setObjectiveFunction(self, function_handle):
        self.objective_function = function_handle

    def callObjectiveFunction(self):
        self.objective_function(self.model)

    # noinspection PyAttributeOutsideInit
    def setVisualizationFunction(self, function_handle, n_iterations=0):
        self.visualization_function = function_handle
        self.visualization_function_iterations = n_iterations

    # noinspection PyAttributeOutsideInit
    def computeSparseMatrix(self):
        self.sparse_matrix = lil_matrix((len(self.residuals), len(self.params)), dtype=int)

        for i, residual in enumerate(self.residuals):
            for param in self.residual_params[residual]:
                idx, _, _, _ = self.params[param]
                self.sparse_matrix[i, idx] = 1

        print('Sparsity matrix:')
        print(pandas.DataFrame(self.sparse_matrix.toarray(), self.residuals, self.params.keys()))

    def printX(self, text='Parameter vector:', x=None):
        if x is None:
            x = self.x

        print(text)
        for i, param_value in enumerate(x):
            for name in self.params:
                idx, _, _, _ = self.params[name]
                if idx == i:
                    print('   ' + name + ' = ' + str(param_value))

    def printModel(self):
        print('Model =\n' + str(self.model))

    def printXAndModel(self, opt):
        self.printX()
        self.printModel()

    def internalObjectiveFunction(self, x):
        """
        A wrapper around the custom given objective function which maps the x vector to the model before calling the
        objetive function and after the call
        :param x: the parameters vector
        """
        self.x = x
        self.fromXToModel()
        error = self.objective_function(self.model)

        if self.counter >= self.visualization_function_iterations:
            self.visualization_function(self.model)
            self.counter = 0
        self.counter += self.counter

        return error

    def startOptimization(self, optimization_options={'x_scale': 'jac', 'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-8,
                                                      'diff_step': 1e0}):
        self.setFirstGuess()
        bounds = (self.bound_min, self.bound_max)
        self.result = least_squares(self.internalObjectiveFunction, self.x, verbose=2, jac_sparsity=self.sparse_matrix,
                                    bounds=bounds, method='trf', args=(), **optimization_options)
        self.xf = list(self.result.x)
        self.finalOptimizationReport()

    def finalOptimizationReport(self):
        """Just print some info and show the images"""
        print('\n-------------\nOptimization finished')
        print(self.result)
        self.printX(text='\nInitial value of parameters', x=self.x0)
        self.printX(text='\nFinal value of parameters', x=self.xf)

        self.fromXToModel(self.xf)
        self.visualization_function(self.model)
        cv2.waitKey(0)
