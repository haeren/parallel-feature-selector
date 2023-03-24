# You can run the program with 4 processes using the following command:
# mpiexec -n 4 python sample.py
import parallel_feature_selector as pfs
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Data must contain numerical values
# Non-numerical columns must be encoded (label/one-hot)
data = pd.read_csv('iris.csv')

# Sklearn estimator
estimator = GaussianNB()

# Feature selection using exhaustive search
pfs.bruteForce(data=data, estimator=estimator)
