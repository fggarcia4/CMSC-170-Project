# KNN
import numpy as np
from DataPreprocessing import TIERS, prepare
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
def distance(X_train, v, p):
    # Minkowski distance formula (based on Lab 06 module formula)
    # d = (sum(|xi - vi|^p))^1/p
    array = np.array(X_train)
    distance = (np.sum(np.abs(array - v) ** p, axis=1)) ** (1/p)
    return distance

def prediction(v,k,p):
    distances = distance(X_train, v, p)
    # find knn
    # creates a list containing the distances
    temp = list(distances)
    nearest = []
    
    for _ in range(k):
        # initial minimum
        minimum = 0 
        for i in range(len(temp)):
            if temp[i] < temp[minimum]:
                minimum = i
        nearest.append(minimum)
        # inf is a number big enough to be bigger than any other, to avoid duplicates
        temp[minimum] = float("inf") 
    
    # creates a dictionary for counting occurences of each tier in nn (TIER reference in DataPreprocessing.py)
    counts = {t: 0 for t in TIERS}
    # dictionary for total distances
    dists = {t: 0.0 for t in TIERS}
    
    for i in nearest:
        counts[Y_train[i]] += 1
        dists[Y_train[i]] += distances[i]
    
    # choose tier that has the most occurences, if tie, choose one with smallest dist
    best_tier = None
    for t in TIERS:
        if counts[t] == 0:
            continue
        if (best_tier is None or (counts[t] > counts[best_tier]) or (counts[t] == counts[best_tier] and dists[t] < dists[best_tier])):
            best_tier = t
    return best_tier

def print_results(tiers, predictions):
    # amount of tests checked
    total = len(tiers)
    
    # ======================================= OFF BY TWO
    # guesses that were correct / matched the actual tier
    correct = 0
    # guesses that were "two-tiers-away"
    off_by_two = 0
    
    for actual, guess in zip(tiers, predictions):
        if actual == guess:
            correct += 1
        # since tiers [low, mid, high] --> 0,1,2, low and high have a difference of 2 == off by two
        elif (actual == "Low" and guess == "High") or (actual == "High" and guess == "Low"):
            off_by_two += 1
        
    # number of incorrect guesses is equal to the total num of tests minus num of correct guesses
    incorrect = total - correct

    # ======================================= MACRO-F1
    f1_scores = []
    for t in TIERS:
        true_positive = 0
        false_positive = 0
        false_negative = 0
    
        for i in range(len(Y_test)):
                actual = Y_test[i]
                predicted = predictions[i]

                if actual == t and predicted == t:
                    true_positive += 1

                elif actual != t and predicted == t:
                    false_positive += 1

                elif actual == t and predicted != t:
                    false_negative += 1

                # calculate precision
                if true_positive + false_positive == 0:
                    precision = 0
                else:
                    precision = true_positive / (true_positive + false_positive)

                # calculate recall
                if true_positive + false_negative == 0:
                    recall = 0
                else:
                    recall = true_positive / (true_positive + false_negative)

                # calculate f1
                if precision + recall == 0:
                    f1 = 0
                else: 
                    f1 = (2 * precision * recall) / (precision + recall)
                
                f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)

    # ======================================= ACCURACY
    accuracy = (correct / total) * 100
    
    
    print("KNN RESULTS")
    print(f"Correct predictions: {correct} ({(correct / total) * 100:.2f}%)")
    print(f"Incorrect predictions: {incorrect} ({(incorrect / total) * 100:.2f}%)")
    print()
    print("Evaluation Metric")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Off-by-two: {off_by_two} ({(off_by_two / total) * 100:.2f}%)")


# tests
dis = distance(X_train, X_test[0], p)
print(len(dis))
prdctn = prediction(X_test[0], k, p)   
print_results(Y_test, [prediction(X_test[i], k, p) for i in range(len(X_test))])