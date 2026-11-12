from DataPreprocessing import prepare
import math

# NAIVE BAYES ALGORITHM
class NaiveBayes:
    def __init__(self): # initialize the naive bayes
        self.tiers = ["Low", "Middle", "High"] # budget tiers
        self.k = 1 # i am using the add one smoothing method as seen on the handout

    # trains the data
    def fit(self, records, labels):

        self.tier_counts = {} # store the number of records in each tier
        self.type_counts = {} # store project type (category) counts for each tier
        self.region_counts = {} # store region counts for each tier
        self.year_means = {} # store the average year for each tier
        self.year_variances = {} # store the year variance for each tier
        self.all_types = set() # store all unique project types
        self.all_regions = set() # store all unique regions

        # iterate through every record
        for record in records:
            self.all_types.add(record["type"]) # unique category add to the list
            self.all_regions.add(record["region"]) # unique region add to the list

        # ------------------------------------------------------------------------
        for budget_tier in self.tiers:
            tier_records = [ # select the records for the current tier
                records[i]
                for i in range(len(labels))
                if labels[i] == budget_tier
            ]

            self.tier_counts[budget_tier] = len(tier_records) # count the training records for this tier
            self.type_counts[budget_tier] = {} # dictionary for project type counts in this tier
            self.region_counts[budget_tier] = {} # dictionary for region counts in this tier

            # print(self.type_counts)

            years = [] # empty list to store the years of this tier

            # iterate through every record in this tier
            for record in tier_records:
                project_type = record["type"] # get project type (category)
                region = record["region"] # get region
                year = record["year"] # get the infrayear

                # count how many times this category appears in the tier
                # get() returns 0 if the project type has not been counted yet
                self.type_counts[budget_tier][project_type] = (
                    self.type_counts[budget_tier].get(project_type, 0) + 1
                )

                # count how many times this region appears in this tier
                # get() returns 0 if the region has not been counted yet
                self.region_counts[budget_tier][region] = (
                    self.region_counts[budget_tier].get(region, 0) + 1
                )

                # add the infrastructure year to the list of years
                years.append(year)

            print(self.type_counts[budget_tier])
            # this is to calculate which years are most seen in the tier
            # ------------------------------------------------------------------------
            mean = sum(years) / len(years) # calculate the average infrastructure year for this class
            variance = sum((year - mean) ** 2 for year in years) / len(years) # calculate the variance to measure how spread out the years are
            self.year_means[budget_tier] = mean # store the calculated average year for this tier
            self.year_variances[budget_tier] = max(variance, 1e-9) # store the variance and prevent it from becoming exactly zero
            # ------------------------------------------------------------------------
        # ------------------------------------------------------------------------

    # GAUSSIAN PROBABILITY
    # this part calculate how likely the year belongs to each tier
    # ------------------------------------------------------------------------
    def gaussian_log_probability(self, year, mean, variance):
        # calculate the Gaussian probability in logarithmic form
        return (
            -0.5 * math.log(2 * math.pi * variance) # calculates the normalization part of gaussian
            - ((year - mean) ** 2) / (2 * variance) # measures how far the year is from the avg year
        )
    # ------------------------------------------------------------------------

    # PREDICTION
    # this function predicts the budget class of one infrastructure project
    # ------------------------------------------------------------------------
    def predict_one(self, record):
        total_records = sum(self.tier_counts.values()) # calculate the total number of training records
        scores = {} # create an empty dictionary to store the scores of each tier

         #calculate the score for each possible budget tier
        for budget_tier in self.tiers:

            # --------------------------------------------------------------
            # check if the current tier has no training records
            if self.tier_counts[budget_tier] == 0:
                # skip if cant be used
                continue
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # calculate the prior probability of the current tier
            # prior = number of records in the class / total training records
            prior = self.tier_counts[budget_tier] / total_records
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # get the number of times the project's type appears in this tier
            # return 0 if the project type has never appeared in this tier
            type_count = self.type_counts[budget_tier].get(
                record["type"], 0
            )
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # calculate the project type probability using Add-One Smoothing
            # https://www.geeksforgeeks.org/nlp/additive-smoothing-techniques-in-language-models/
            type_probability = (type_count + self.k) / (self.tier_counts[budget_tier] + self.k * len(self.all_types))
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # get the number of times the project's region appears in this tier
            # return 0 if the region has never appeared in this tier
            region_count = self.region_counts[budget_tier].get(
                record["region"], 0
            )
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # calculate the region probability using Add-One Smoothing
            region_probability = (region_count + self.k) / (self.tier_counts[budget_tier] + self.k * len(self.all_regions))
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # calculate the Gaussian log probability of the infrastructure year
            year_probability = self.gaussian_log_probability(
                record["year"], # year of the project being predicted
                self.year_means[budget_tier], # average year of the current tier
                self.year_variances[budget_tier] # year variance of the current tier
            )
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # combine the probabilities using logarithms
            scores[budget_tier] = (
                math.log(prior)
                + math.log(type_probability)
                + math.log(region_probability)
                + year_probability
            )
            # --------------------------------------------------------------

        # return the tier with the highest score
        return max(scores, key=scores.get)
        # ------------------------------------------------------------------------


# MAIN PROGRAM
data = prepare()

# --------------------------------------------------------------
X_train = data["train_records"]
X_test = data["test_records"]
Y_train = data["y_train"]
Y_test = data["y_test"]
# --------------------------------------------------------------

# --------------------------------------------------------------
print("\nNAIVE BAYES DATA")
print("-------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
# --------------------------------------------------------------

model = NaiveBayes()

model.fit(X_train, Y_train)

predictions = [
    model.predict_one(record)
    for record in X_test
]

correct = 0

for i in range(len(Y_test)):
    if predictions[i] == Y_test[i]:
        correct += 1

# EVALUATION METRIC COMPUTATION
tiers = ["Low", "Middle", "High"]
# --------------------------------------------------------------
# Macro-F1
f1_scores = []
for budget_tier in tiers:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(Y_test)):
        actual = Y_test[i]
        predicted = predictions[i]

        if actual == budget_tier and predicted == budget_tier:
            true_positive += 1

        elif actual != budget_tier and predicted == budget_tier:
            false_positive += 1

        elif actual == budget_tier and predicted != budget_tier:
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
# --------------------------------------------------------------

# --------------------------------------------------------------
# Accuracy
accuracy = (correct / len(Y_test)) * 100
# --------------------------------------------------------------

# --------------------------------------------------------------
# Off-by-two Rate
off_by_two = 0

for i in range(len(Y_test)):
    actual = Y_test[i]
    predicted = predictions[i]

    actual_position = tiers.index(actual)
    predicted_position = tiers.index(predicted)

    difference = abs(actual_position - predicted_position)

    if difference == 2:
        off_by_two += 1

off_by_two_rate = off_by_two / len(Y_test)
# --------------------------------------------------------------

# --------------------------------------------------------------
print("\nNAIVE BAYES RESULTS")
print("-------------------------")
print("Correct predictions:", correct)
print("Incorrect predictions:", len(Y_test) - correct)
# --------------------------------------------------------------

# --------------------------------------------------------------
print("\nEVALUATION METRIC")
print("Macro-F1 Score: ", round(macro_f1, 4))
print("Accuracy: ", round(accuracy, 2), "%")
print("Off-by-two Rate: ", round(off_by_two_rate * 100, 2), "%")
# --------------------------------------------------------------

# --------------------------------------------------------------
print("\nSAMPLE PREDICTION")
print("-------------------------")
print("Project:", X_test[2000])
print("Actual tier:", Y_test[2000])
print("Predicted tier:", predictions[2000])
print("\n")
print("Project:", X_test[1200])
print("Actual tier:", Y_test[1200])
print("Predicted tier:", predictions[1200])
print("\n")
print("Project:", X_test[2500])
print("Actual tier:", Y_test[2500])
print("Predicted tier:", predictions[2500])
print("\n")
print("Project:", X_test[107])
print("Actual tier:", Y_test[107])
print("Predicted tier:", predictions[107])
print("\n")
print("Project:", X_test[2050])
print("Actual tier:", Y_test[2050])
print("Predicted tier:", predictions[2050])
# --------------------------------------------------------------
