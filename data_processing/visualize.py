import argparse
import os
import numpy as np
import matplotlib
# Force headless backend for SLURM
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

import torch

from typing import Tuple, Dict
import os, glob, re, time
import pickle
# Own imports
try:
    from data_processing.sample import Sample
    from data_processing.utils import iter_results_db
except ImportError:
    from sample import Sample
    from utils import iter_results_db
