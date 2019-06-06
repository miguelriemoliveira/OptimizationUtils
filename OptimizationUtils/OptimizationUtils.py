# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import matplotlib
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


# ------------------------
# FUNCTION DEFINITION
# ------------------------
def addArguments(ap):
    """ Adds to an argument parser struct the list of command line arguments necessary or used by the Optimization utils

    :param ap:
    :return:
    """
    ap.add_argument("-sv", "--skip_vertices", help="skip vertices. Useful for fast testing", type=int, default=1)
    ap.add_argument("-z", "--z_inconsistency_threshold", help="threshold for max z inconsistency value", type=float,
                    default=0.05)
    ap.add_argument("-vpv", "--view_projected_vertices", help="visualize projections of vertices onto images",
                    action='store_true', default=False)
    ap.add_argument("-vo", "--view_optimization", help="...", action='store_true', default=False)
    return ap


# -------------------------------------------------------------------------------
# CLASS
# -------------------------------------------------------------------------------
class Optimizer:

    def __init__(self):
        """
        Initializes class properties with default values.
        """

        self.data_models = {}  # a dict with a set of variables or structures to be used by the objective function
        self.groups = OrderedDict()  # groups of params an ordered dict where key={name} and value = namedtuple('ParamT')

        self.x = []  # a list of floats (the actual parameters)
        self.x0 = []  # the initial value of the parameters
        self.xf = []  # the final value of the parameters

        self.residuals = OrderedDict()  # ordered dict: key={residual} value = [params that influence this residual]
        self.sparse_matrix = None
        self.result = None  # to contain the optimization result
        self.objective_function = None  # to contain the objective function
        # self.visualization_function = None
        self.first_call_of_objective_function = True

        # Visualization stuff
        self.vis_function_handle = None  # to contain a handle to the visualization function
        self.vis_niterations = 0
        self.vis_counter = 0

    # ---------------------------
    # Optimizer configuration
    # ---------------------------
    def addModelData(self, name, data):
        """ Should be a dictionary containing every static data to be used by the cost function
        :param name: string containing the name of the data
        :param data: object containing the data
        """
        if name in self.data_models:  # Cannot add a parameter that already exists
            raise ValueError('Data ' + name + ' already exits in model dict.')
        else:
            self.data_models[name] = data
            # print('Added data ' + name + ' to model dict.')

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

        if not data_key in self.data_models:
            raise ValueError('Dataset ' + data_key + ' does not exist. Cannot add group ' + group_name + '.')

        value = getter(self.data_models[data_key])
        if not type(value) is list or not (len(value) == 1):
            raise ValueError('For scalar parameters, getter must return a list of lenght 1. Returned list = ' + str(
                value) + ' of type ' + str(type(value)))

        param_names = [group_name]  # a single parameter with the same name as the group
        idx = [len(self.x)]
        self.groups[group_name] = ParamT(param_names, idx, data_key, getter, setter, [bound_max],
                                         [bound_min])  # add to group dict
        self.x.append(value[0])  # set initial value in x using the value from the data model
        # print('Pushed scalar param ' + group_name + ' to group ' + group_name)

    def pushParamV3(self, group_name, data_key, getter, setter, bound_max=(+inf, +inf, +inf),
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

        if not data_key in self.data_models:  # Check if we have the data_key in the data dictionary
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
        values = getter(self.data_models[data_key])
        for value in values:
            self.x.append(value)  # set initial value in x
        # print('Pushed translation group ' + group_name + ' with params ' + str(param_names))

    def pushResidual(self, name, params=None):
        """Adds a new residual to the existing list of residuals

        :param name: name of residual
        :type name: string
        :param params: parameter names which affect this residual
        :type params: list
        """
        # TODO check if all listed params exist in the self.params
        self.residuals[name] = params

    def setObjectiveFunction(self, handle):
        # type: (function) -> object
        """Provide a pointer to the objective function

        :param handle: the function handle
        """
        self.objective_function = handle

    def setVisualizationFunction(self, handle, always_visualize, niterations=0):
        """ Sets up the visualization function to be called to plot the data during the optimization procedure.

        :param handle: handle to the function
        :param always_visualize: call visualization function during optimization or just at the end
        :param niterations: number of iterations at which the visualization function is called.
        """

        self.vis_function_handle = handle
        self.vis_niterations = niterations
        self.always_visualize = always_visualize

    # ---------------------------
    # Optimization methods
    # ---------------------------
    def callObjectiveFunction(self):
        """ Just an utility to call the objective function once. """
        return self.internalObjectiveFunction(self.x)

    def internalObjectiveFunction(self, x):
        # type: (list) -> list
        """ A wrapper around the custom given objective function which maps the x vector to the model before calling the
        objective function and after the call

        :param x: the parameters vector
        """
        self.x = x  # setup x parameters.
        self.fromXToData()  # Copy from parameters to data models.
        errors = self.objective_function(self.data_models)  # Call objective func. with updated data models.

        # Visualization: skip if counter does not exceed blackout interval
        if self.always_visualize and self.vis_counter >= self.vis_niterations:
            self.vis_counter = 0  # reset counter
            self.vis_function_handle(self.data_models)  # call visualization function
            self.plot_handle.set_data(range(0, len(errors)), errors)  # redraw residuals plot
            self.ax.relim()  # recompute new limits
            self.ax.autoscale_view()  # re-enable auto scale
            self.wm.waitForKey(time_to_wait=0.01, verbose=True)  # wait a bit

            # Printing information
            self.printParameters(flg_simple=True)
            self.printResiduals(errors)
            print('\nAverage error = ' + str(np.average(errors)) + '\n')

        else:
            self.vis_counter += 1

        return errors

    def startOptimization(self, optimization_options={'x_scale': 'jac', 'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8,
                                                      'diff_step': 1e-4}):
        """ Initializes the optimization procedure.

        :param optimization_options: dict with options for the least squares scipy function.
        Check https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        """
        self.x0 = deepcopy(self.x)  # store current x as initial parameter values
        self.fromXToData()  # copy from x to data models
        self.errors0 = self.objective_function(self.data_models)  # call obj. func. (once) to get initial residuals
        errors = self.objective_function(self.data_models)  # Call objective func. with updated data models.

        # Setup boundaries for parameters
        bounds_min = []
        bounds_max = []
        for name in self.groups:
            _, _, _, _, _, bound_max, bound_min = self.groups[name]
            bounds_max.extend(bound_max)
            bounds_min.extend(bound_min)

        if self.always_visualize:
            self.drawResidualsFigure()  # First draw of residuals figure
            self.vis_counter = 0  # reset counter
            self.vis_function_handle(self.data_models)  # call visualization function
            self.plot_handle.set_data(range(0, len(errors)), errors)  # redraw residuals plot
            self.ax.relim()  # recompute new limits
            self.ax.autoscale_view()  # re-enable auto scale
            self.wm.waitForKey(time_to_wait=0.01, verbose=True)  # wait a bit

            # Printing information
            self.printParameters(flg_simple=True)
            self.printResiduals(errors)
            print('\nAverage error = ' + str(np.average(errors)) + '\n')
            self.wm.waitForKey(time_to_wait=None, verbose=True)  # wait a bit

        # Call optimization function (finally!)
        print("\n\nStarting optimization")
        self.result = least_squares(self.internalObjectiveFunction, self.x, verbose=2, jac_sparsity=self.sparse_matrix,
                                    bounds=(bounds_min, bounds_max), method='trf', args=(), **optimization_options)

        self.xf = deepcopy(list(self.result.x))  # Store final x values
        self.finalOptimizationReport()  # print an informative report

    def finalOptimizationReport(self):
        """Just print some info and show the images"""
        print('\n-------------\nOptimization finished')
        print(self.result)
        # self.printX(preamble_text='\nInitial value of parameters', x=self.x0)
        # self.printX(preamble_text='\nFinal value of parameters', x=self.xf)

        self.fromXToData(self.xf)

        if self.always_visualize:
            self.vis_function_handle(self.data_models)
            self.wm.waitForKey(time_to_wait=None, verbose=False)

    # ---------------------------
    # Utilities
    # ---------------------------
    def addNoiseToX(self, noise=0.1, x=None):
        """ Adds uniform noise to the values in the parameter vector x

        :param noise: magnitude of the noise. 0.1 would generate a value from 0.9 to 1.1 of the current value
        :param x: parameter vector. If None the currently stored in the class is used.
        :return: new parameter vector with noise.
        """
        # type: (float, list) -> list
        if x is None:
            x = self.x

        return x * np.array([random.uniform(1 - noise, 1 + noise) for _ in xrange(len(x))], dtype=np.float)

    def getParameters(self):
        """ Gets all the existing parameters

        :return: a list with all the parameter names.
        """
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

    def fromDataToX(self, x=None):
        """ Copies values of all parameters from the data to the vector x

        :param x:  parameter vector. If None the currently stored in the class is used.
        """
        if x is None:
            x = self.x

        for group_name, group in self.groups.items():
            values = group.getter(self.data_models[group.data_key])
            for i, idx in enumerate(group.idx):
                x[idx] = values[i]

    def fromXToData(self, x=None):
        """ Copies values of all parameters from vector x to the data

        :param x:  parameter vector. If None the currently stored in the class is used.
        """
        if x is None:
            x = self.x

        for group_name, group in self.groups.items():
            values = []
            for idx in group.idx:
                values.append(x[idx])

            group.setter(self.data_models[group.data_key], values)

    def computeSparseMatrix(self):
        """ Computes the sparse matrix given the parameters and the residuals. Should be called only after setting both.

        """
        params = self.getParameters()
        self.sparse_matrix = lil_matrix((len(self.residuals), len(params)), dtype=int)

        for i, key in enumerate(self.residuals):
            for param in self.residuals[key]:
                # print("param = " + param)
                for group_name, group in self.groups.items():
                    if param in group.param_names:
                        idx_in_group = group.param_names.index(param)
                        # print("param_names = " + str(group.param_names))
                        idx = group.idx[idx_in_group]
                        # print("group.idx = " + str(group.idx))
                        # print("idx_in_group = " + str(idx_in_group))
                        # print("idx = " + str(idx))
                        self.sparse_matrix[i, idx] = 1

        print('Sparsity matrix:')
        data_frame = pandas.DataFrame(self.sparse_matrix.toarray(), self.residuals, params)
        print(data_frame)
        data_frame.to_csv('sparse_matrix.csv')

    # ---------------------------
    # Print and display
    # ---------------------------
    def printX(self, x=None):
        """ Prints the list of parameters

        :param x: list of parameters. If None prints the currently stored list.
        """
        if x is None:
            x = self.x

        for group_name, group in self.groups.items():
            print('Group ' + str(group_name) + ' has parameters:')
            values_in_data = group.getter(self.data_models[group.data_key])
            for i, param_name in enumerate(group.param_names):
                print('--- ' + str(param_name) + ' = ' + str(values_in_data[i]) + ' (in data) ' + str(
                    x[group.idx[i]]) + ' (in x)')

        print(self.x)

    def printParameters(self, x=None, flg_simple=False, text=None):
        """ Prints the current values of the parameters in the parameter list as well as the corresponding data
        models.

        :param x: list of parameters. If None prints the currently stored list.
        :param flg_simple:
        :param text: string to write as a header for the table of parameter values
        """
        if x is None:
            x = self.x

        # Build a panda data frame and then print a nice table
        rows = []  # get a list of parameters
        table = []
        for group_name, group in self.groups.items():
            values_in_data = group.getter(self.data_models[group.data_key])
            for i, param_name in enumerate(group.param_names):
                rows.append(param_name)
                table.append([group_name, x[group.idx[i]], values_in_data[i]])

        if text is None:
            print('\nParameters:')
        else:
            print(text)


        df = pandas.DataFrame(table, rows, ['Group', 'x', 'data'])
        if flg_simple:
            # https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c
            print(df[['x']])
        else:
            print(df)

    def printModelsInfo(self):
        """ Prints information about the currently configured models """
        print('There are ' + str(len(self.data_models)) + ' data models stored: ' + str(self.data_models))

    def printXAndModelsInfo(self):
        """ Just a wrapper of two other methods. """
        self.printX()
        self.printModelsInfo()

    def printResiduals(self, errors=None):
        """ Prints the current values of the residuals.

        :param errors:
        """
        rows = []  # get a list of residuals
        table = []
        if errors is None:
            errors = np.full((len(self.residuals)), np.nan)
            # errors=np.nans((len(self.residuals)))

        for i, residual in enumerate(self.residuals):
            rows.append(residual)
            table.append(errors[i])

        print('\nResiduals:')
        df = pandas.DataFrame(table, rows, ['error'])
        print(df)

    # ---------------------------
    # Drawing and figures
    # ---------------------------
    def drawResidualsFigure(self):

        # Prepare residuals figure
        self.figure_residuals = matplotlib.pyplot.figure()
        self.wm = KeyPressManager.KeyPressManager.WindowManager(self.figure_residuals)
        self.ax = self.figure_residuals.add_subplot(1, 1, 1)
        x = range(0, len(self.errors0))
        self.initial_residuals_handle, = self.ax.plot(x, self.errors0, color='green', marker='o',
                                                      linestyle='solid', linewidth=2, markersize=6)
        self.ax.plot(x, [0] * len(self.errors0), color='black', linestyle='dashed', linewidth=2, markersize=6)
        self.ax.set_xticks(x, minor=False)
        self.ax.set_xticks([], minor=True)
        self.ax.set_xticklabels(list(self.residuals.keys()))

        matplotlib.pyplot.title('Optimization Residuals')
        matplotlib.pyplot.xlabel('Residuals')
        matplotlib.pyplot.ylabel('Values')
        for tick in self.ax.get_xticklabels():
            tick.set_rotation(90)

        self.wm.waitForKey(time_to_wait=0.01, verbose=True)

        self.plot_handle, = self.ax.plot(range(0, len(self.errors0)), self.errors0, color='blue', marker='s',
                                         linestyle='solid', linewidth=2, markersize=6)
        matplotlib.pyplot.legend((self.initial_residuals_handle, self.plot_handle), ('Initial', 'Current'))
        self.ax.relim()
        self.ax.autoscale_view()
        self.wm.waitForKey(time_to_wait=0.01, verbose=True)
