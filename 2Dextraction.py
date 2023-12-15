# -*- coding: utf-8 -*-
"""
Simple usage example for PyReduce
Loads a sample UVES dataset, and runs the full extraction
"""
import os.path

import pyreduce
from pyreduce.configuration import get_configuration_for_instrument
from pyreduce.instruments.common import create_custom_instrument
from pyreduce.reduce import Reducer
from pyreduce.util import start_logging

# Define the path to support files if possible
# otherwise set them to None
# Obviously they are necessary for their respective steps
bpm_mask = None #"path/to/bpm_mask.fits"
wavecal_file = None #"path/to/wavecal_file"


# create our custom instrument
instrument = create_custom_instrument(
    "custom", extension=0, mask_file=bpm_mask, wavecal_file=wavecal_file
)

# Override default values
# those can either be fixed values or refer to FITS header keywords
instrument.info["readnoise"] = 1
instrument.info["prescan_x"] = 0 # "PRESCAN X"
instrument.info["extension"] = 0 # 

# For loading the config we specify pyreduce as the source, since this is the default
config = get_configuration_for_instrument("pyreduce", plot=1)
# Define your own configuration
config["orders"]["degree"] = 6

# We define the path to the output directory
input_dir = './data/test_extract/'
output_dir = input_dir + "reduced"

# Since we can't find the files ourselves (at least not without defining the criteria we are looking for)
# We need to manually define which files go where
#change this to a sort files function
files = {"bias": [], 
         "science": [input_dir + "HISPEC_HK_test_8.5deg.fits"],
         "orders": [input_dir + "HISPEC_HK_test_8.5deg.fits"],
         "flat": [],
         "wave": []}


# (optional) We need to define the log file
log_file = "./logs/log_file.txt"
start_logging(log_file)

# Define other parameter for PyReduce
target = ""
night = "2022-08-12"
mode = ""
steps = (
    #"bias",
    #"flat",
    "orders",
    #"curvature",
    #"scatter",
    #"norm_flat",
    # "wavecal",
    # "freq_comb",
    "science",
    #"continuum",
    # "finalize",
)

# Call the PyReduce algorithm
reducer = Reducer(
    files,
    output_dir,
    target,
    instrument,
    mode,
    night,
    config,
    # order_range=order_range,
    # skip_existing=False,
)

data = reducer.run_steps(steps=steps)



