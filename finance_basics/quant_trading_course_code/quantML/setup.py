
import mysql.connector
import  pandas as pd
import numpy as np
import scipy
from datetime import  datetime

pd.options.mode.chained_assignment=None

from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.preprocessing import Imputer

from xgboost.sklearn import XGBClassifier

from sklearn.neighbors import  KNeighborsRegressor
