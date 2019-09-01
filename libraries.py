"""
@author: payonear
"""
# import necessary libraries
import pandas as pd
import numpy as np
import math
from scipy.stats import kurtosis
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from matplotlib.backends.backend_pdf import PdfPages
import os
from sklearn import metrics as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)