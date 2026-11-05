# KNN
import numpy as np
from DataPreprocessing import TIERS, prepare

# k nearest neighbor
# get the "nearest" using the distance
class KNN:
    def __init__(self, k, p): # initialize KNN
        self.k = k # number of neighbors to look at, odd number to avoid ties
        self.p = p # Minkowski distance parameter

    # this is to store training data (lazy learning)
    def training_data(self, X_train, y_train):
        self.X_train = np.asarray(X_train) # features: type, region, year
        self.y_train = y_train # tiers: low, middle, high

    # for each test point, get the distances, sort, then get k nearest
    def predict(self, X_test):
        # store predictions for each test point
        predictions = []
        # for each test point v in x_test, (this will run for each test point)
        for v in X_test:
            # get distance for each test point v
            distances = self.distance(v)
            # get distances from nearest to farthest (until k nearest only, [:self.k])
            nearest_distances = np.argsort(distances)[:self.k]
            # count tiers
            # get the tiers of the k nearest neighbors, 
            nearest_tiers = []
            # for each index i,
            for i in nearest_distances:
                # append y_train[i]
                nearest_tiers.append(self.y_train[i])
                
            # using majority count fxn, use that to predict test point's label
            predicted_tier = self.majority_count(nearest_tiers)
            # append predicted tier to predictions list
            predictions.append(predicted_tier)
        return predictions

    # compute distance between a test point and all other training points
    def distance(self,  v):
        # Minkowski distance formula (based on Lab 06 module formula)
        # d = (sum(|xi - vi|^p))^1/p
        distance = np.sum(np.abs(self.X_train - v) ** self.p, axis=1) ** (1 / self.p)
        return distance

    def majority_count(self, tiers):
        # initialize dictionary count
        counts = {}
        for t in tiers:
            # find tier in dictionary, if not found, set to 0, then add 1 to count
            counts[t] = counts.get(t, 0) + 1
            
        return max(counts, key=counts.get)


# =====================================================================================================================
# GET k AND p FROM USER
# rule: k is odd, p >= 1
# created functions for getting k and p from user input, with validation
def get_k(train):
    while True:
        try:
            k = int(input("Enter an odd value of k: ")) # prompt user for k
        except ValueError: 
            # if non-integer, prompt again
            print("Invalid input. Please enter an integer value.")
            continue
        # if k is not positive, odd integer + greater or equal to training size, prompt again
        if k > 0 and k % 2 == 1 and k <= train:
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
        for i in range(len(tiers)):
            actual = tiers[i]
            predicted = predictions[i]
            if actual == t and predicted == t:
                true_positive += 1
            elif actual != t and predicted == t:
                false_positive += 1
            elif actual == t and predicted != t:
                false_negative += 1
        # calculate precision =======================================
        if true_positive + false_positive == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)
        # calculate recall =======================================
        if true_positive + false_negative == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)
        # calculate f1 =======================================
        if precision + recall == 0:
            f1 = 0
        else: 
            f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / len(f1_scores)
    # ======================================= ACCURACY
    accuracy = (correct / total) * 100
    # ======================================= PRINTTT
    print("KNN RESULTS")
    print(f"Correct predictions: {correct} ({(correct / total) * 100:.2f}%)")
    print(f"Incorrect predictions: {incorrect} ({(incorrect / total) * 100:.2f}%)")
    print()
    print("Evaluation Metric")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Off-by-two: {off_by_two} ({(off_by_two / total) * 100:.2f}%)")

# ===================================================================================================================== 
if __name__ == "__main__":
    # prepare data
    data = prepare()
    
    # show output
    print("Dataset loaded successfully.")
    # data["records"] contains all records, with that, you can access its length
    print(f"Total number of records: {len(data['records'])}")
    # since len(x_train) == len(y_train), we can use either to get the number of training instances
    print(f"Training instances: {len('X_train')}")
    print(f"Testing instances : {len('X_test')}")
    
    k = get_k(len(data["X_train"]))
    p = get_p()

    # initialize knn
    knn = KNN(k,p)
    knn.training_data(data["X_train"], data["y_train"])

    # limit test points (para di mabagal yng pagload,,, comment out or use None to use all data)
    limit = 2000
    
    # predict using knn
    predictions = knn.predict(data["X_test"][:limit]) # use limit,, para from start, to limit lang muna

    print_results(data["y_test"][:limit], predictions)