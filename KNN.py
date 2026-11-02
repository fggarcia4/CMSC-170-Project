# KNN
import numpy as np
from DataPreprocessing import prepare
import csv

# LOADS DATA
# prepare cleans and splits train & test and encode features
data = prepare()
X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["y_train"]
Y_test = data["y_test"]
#============================================
# show output
print("Dataset loaded successfully.")
# data["records"] contains all records, with that, you can access its length
print(f"Total number of records: {len(data['records'])}")
# since len(x_train) == len(y_train), we can use either to get the number of training instances
print(f"Training instances: {len(X_train)}")
print(f"Testing instances : {len(X_test)}")
#============================================
# GET k AND p FROM USER
# rule: k is odd, p >= 1
# created functions for getting k and p from user input, with validation
def get_k():
    while True:
        try:
            k = int(input("Enter an odd value of k: ")) # prompt user for k
        except ValueError: 
            # if non-integer, prompt again
            print("Invalid input. Please enter an integer value.")
            continue
        # if k is not positive, odd integer + greater or equal to training size, prompt again
        if k > 0 and k % 2 == 1 and k <= len(X_train):
            return k
        print("k must be a positive odd integer.")
# same comments from get_k, this is tailored specifically for p -> a float that is greater than or equal to 1
def get_p():
    while True:
        try:
            p = float(input("Enter value of p for Minkowski distance: ")) # prompt
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if p >= 1:
            return p
        print("p must be at least 1.")    
#============================================
k = get_k()
p = get_p()
#============================================
# for testing only
# TEST = 1000
#============================================#============================================#============================================
# COMPUTE FOR ACCURACY
def minkowski(X_train,v,p):
  total_sum = 0
  for f in features:
    # this will get absolute value of the difference of each feature
    distance = abs(float(x[f])-float(v[f]))
    # add to total sum the distance^p
    total_sum += distance**p
  # use the total sum^1/p to win
  dis_minkowski = total_sum**(1/p)

  return dis_minkowski

def accuracy():
  print("accuracy")

print(minkowski(training[0],testing[0],p))