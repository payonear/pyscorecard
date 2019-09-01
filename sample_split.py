
"""
@author: payonear
"""

from functions import *

# import base_coarse Data Set (put the right directory)
base_coarse = pd.read_csv('PATH.../base_coarse.csv')

X = base_coarse.iloc[:,:-1]
y = base_coarse.iloc[:,-1]
# stratified split on training and test sample
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1916, stratify = y)

train = pd.merge(y_train, X_train, left_index = True, right_index = True)
test = pd.merge(y_test, X_test, left_index = True, right_index = True)

train.to_csv('train.csv', index = None, header = True)
test.to_csv('test.csv', index = None, header = True)