#!/usr/bin/env python
"""
This is a long, multiline description
"""

#-------------------------------------------------------------------------------
#--- Add rgbd_tm module to the sys.path
#--- Will probably have to be done for the code which is using this API
#-------------------------------------------------------------------------------
#import sys
#import os.path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#-------------------------------------------------------------------------------
#--- IMPORTS (standard, then third party, then my own modules)
#-------------------------------------------------------------------------------
import argparse  #to read command line arguments

import OptimizationUtils.OptimizationUtils as OU

#-------------------------------------------------------------------------------
#--- HEADER
#-------------------------------------------------------------------------------
__author__ = "Miguel Riem de Oliveira"
__date__ = "2018"
__copyright__ = "Miguel Riem de Oliveira"
__credits__ = ["Miguel Riem de Oliveira"]
__license__ = ""
__version__ = "1.0"
__maintainer__ = "Miguel Oliveira"
__email__ = "m.riem.oliveira@gmail.com"
__status__ = "Development"

#-------------------------------------------------------------------------------
#--- FUNCTIONS
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#--- MAIN
#-------------------------------------------------------------------------------
if __name__ == "__main__":

    #---------------------------------------
    #--- Parse command line argument
    #---------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path_to_images", help = "path to the folder that contains the images", required=True)
    args = vars(ap.parse_args())

    #---------------------------------------
    #--- INITIALIZATION
    #---------------------------------------
    optimizer = OU.Optimizer()



    #---------------------------------------
    #--- USING THE USER INTERFACE
    #---------------------------------------


    #---------------------------------------
    #--- USING THE API
    #---------------------------------------


