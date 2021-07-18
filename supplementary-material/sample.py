import parallel_feature_selector as pfs
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Data must contain numerical values
# Non-numerical columns must be encoded (label/one-hot)
data = pd.read_csv('iris.csv')

# Sklearn estimator
estimator = GaussianNB()

pfs.bruteForce(data=data, estimator=estimator)