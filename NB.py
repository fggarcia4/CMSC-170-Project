from DataPreprocessing import prepare
import math


# --------------------------------------------------------------
# NAIVE BAYES ALGORITHM
# --------------------------------------------------------------

class NaiveBayes:

    def __init__(self):

        # eight input features
        self.features = [
            "category",
            "componentCategories",
            "region",
            "province",
            "contractor",
            "infraYear",
            "programName",
            "sourceOfFunds"
        ]

        # 2 types of features used
        # numeric (digits) and categorical (characters)
        self.numeric_features = ["infraYear"] # infraYear is the only numeric feature aside for target (budget)

        # categorical features
        self.categorical_features = [
            feature for feature in self.features
            if feature not in self.numeric_features
        ]

        # tiers
        # low, middle, high
        self.tiers = ["Low", "Middle", "High"]

        # add-one smoothing to prevent 0 probability
        self.k = 1


    # ----------------------------------------------------------
    # CONVERT FEATURE VALUES
    # ----------------------------------------------------------

    # converts all missing values into unknown

    def categorical_value(self, value):

        if value is None or str(value).strip() == "":
            return "Unknown"

        return str(value).strip()


    # convert digits to float
    # will help in solving gaussian for decimals
    def numeric_value(self, value):

        try:
            if value is None or str(value).strip() == "":
                return None

            return float(value)

        except (ValueError, TypeError):
            return None


    # ----------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------

    def fit(self, records, labels):

        # count how many training records belong to each tier
        self.tier_counts = {}

        # counts category occurences per tier
        self.categorical_counts = {}

        # store all possible categorical values
        self.all_categories = {}

        # Store the mean and variance of infraYear
        self.mean = {}
        self.variance = {}


        # ------------------------------------------------------
        # GET ALL POSSIBLE CATEGORICAL VALUES
        # ------------------------------------------------------

        for feature in self.categorical_features:

            self.all_categories[feature] = set()

            for record in records:

                value = self.categorical_value(
                    record.get(feature)
                )

                self.all_categories[feature].add(value)


        # ------------------------------------------------------
        # CALCULATE VALUES FOR EACH BUDGET TIER
        # ------------------------------------------------------

        for budget_tier in self.tiers:

            # get all records that belong to the current tier
            tier_records = [
                records[i]
                for i in range(len(labels))
                if labels[i] == budget_tier
            ]

            # count the records in the current tier
            self.tier_counts[budget_tier] = len(tier_records)

            self.categorical_counts[budget_tier] = {}
            self.mean[budget_tier] = {}
            self.variance[budget_tier] = {}


            # --------------------------------------------------
            # COUNT CATEGORICAL FEATURES
            # --------------------------------------------------

            for feature in self.categorical_features:

                self.categorical_counts[budget_tier][feature] = {}

                for record in tier_records:

                    value = self.categorical_value(
                        record.get(feature)
                    )

                    # add one to the count of the current value
                    self.categorical_counts[budget_tier][feature][value] = (
                        self.categorical_counts[budget_tier][feature].get(value, 0) + 1
                    )


            # --------------------------------------------------
            # CALCULATE MEAN AND VARIANCE OF infraYear
            # --------------------------------------------------

            for feature in self.numeric_features:

                values = []

                for record in tier_records:

                    value = self.numeric_value(
                        record.get(feature)
                    )

                    if value is not None:
                        values.append(value)

                if len(values) > 0:

                    # calculate the average year
                    mean = sum(values) / len(values)

                    # calculate the variance
                    variance = sum(
                        (value - mean) ** 2 for value in values
                    ) / len(values)

                    self.mean[budget_tier][feature] = mean

                    # prevent variance from becoming zero.
                    self.variance[budget_tier][feature] = max(
                        variance, 1e-9
                    )

                else:

                    # Use safe default values if no valid year exists.
                    self.mean[budget_tier][feature] = 0
                    self.variance[budget_tier][feature] = 1


    # ----------------------------------------------------------
    # GAUSSIAN PROBABILITY
    # ----------------------------------------------------------

    def gaussian_log_probability(self, value, mean, variance):

        return (
            -0.5 * math.log(2 * math.pi * variance)
            - ((value - mean) ** 2) / (2 * variance)
        )


    # ----------------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------------

    def predict_one(self, record):

        # get the total number of training records
        total_records = sum(self.tier_counts.values())

        scores = {}

        # calculate the probability for each budget tier
        for budget_tier in self.tiers:

            # skip a tier if it has no training records
            if self.tier_counts[budget_tier] == 0:
                continue

            # calculate the prior probability.
            prior = self.tier_counts[budget_tier] / total_records

            score = math.log(prior)


            # --------------------------------------------------
            # CATEGORICAL FEATURE PROBABILITIES
            # --------------------------------------------------

            for feature in self.categorical_features:

                value = self.categorical_value(
                    record.get(feature)
                )

                # get the number of times this value appeared in the current budget tier
                value_count = self.categorical_counts[budget_tier][feature].get(
                    value, 0
                )

                # get the number of possible values for this feature
                size = len(self.all_categories[feature])

                # apply add-one smoothing.
                probability = (value_count + self.k) / (self.tier_counts[budget_tier] + self.k * (size + 1))

                score += math.log(probability)


            # --------------------------------------------------
            # infraYear PROBABILITY
            # --------------------------------------------------

            for feature in self.numeric_features:

                value = self.numeric_value(
                    record.get(feature)
                )

                # skip if its value is missing
                if value is not None:

                    mean = self.mean[budget_tier][feature]

                    variance = self.variance[budget_tier][feature]

                    score += self.gaussian_log_probability(value, mean, variance)


            # store the final score for the current tier
            scores[budget_tier] = score

        # return the tier with the highest probability score
        return max(scores, key=scores.get)


# --------------------------------------------------------------
# EVALUATION
# --------------------------------------------------------------

def evaluate_model(actual_tiers, predictions):

    tiers = ["Low", "Middle", "High"]

    correct = 0

    for i in range(len(actual_tiers)):

        if predictions[i] == actual_tiers[i]:
            correct += 1


    # ----------------------------------------------------------
    # MACRO-F1 SCORE
    # ----------------------------------------------------------

    f1_scores = []

    for budget_tier in tiers:

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for i in range(len(actual_tiers)):

            actual = actual_tiers[i]
            predicted = predictions[i]

            if actual == budget_tier and predicted == budget_tier:
                true_positive += 1

            elif actual != budget_tier and predicted == budget_tier:
                false_positive += 1

            elif actual == budget_tier and predicted != budget_tier:
                false_negative += 1


        # precision
        if true_positive + false_positive == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)


        # recall
        if true_positive + false_negative == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)


        # F1 score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        f1_scores.append(f1)


    # get the average F1 score of all three tiers.
    macro_f1 = sum(f1_scores) / len(f1_scores)


    # ----------------------------------------------------------
    # ACCURACY
    # ----------------------------------------------------------

    accuracy = (correct / len(actual_tiers)) * 100


    # ----------------------------------------------------------
    # OFF-BY-TWO RATE PER TIER
    # ----------------------------------------------------------

    off_by_two = {
        "Low": 0,
        "Middle": 0,
        "High": 0
    }

    tier_totals = {
        "Low": 0,
        "Middle": 0,
        "High": 0
    }

    for i in range(len(actual_tiers)):

        actual = actual_tiers[i]
        predicted = predictions[i]

        # count the total actual projects for each tier
        tier_totals[actual] += 1

        actual_position = tiers.index(actual)
        predicted_position = tiers.index(predicted)

        difference = abs(actual_position - predicted_position)

        # count errors that are two tiers away
        # the error is assigned to the actual tier
        if difference == 2:
            off_by_two[actual] += 1


    # calculate the off-by-two rate for each tier
    off_by_two_rate = {}

    for budget_tier in tiers:

        if tier_totals[budget_tier] == 0:
            off_by_two_rate[budget_tier] = 0
        else:
            off_by_two_rate[budget_tier] = (
                off_by_two[budget_tier] / tier_totals[budget_tier]
            ) * 100


    return {
        "correct": correct,
        "incorrect": len(actual_tiers) - correct,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "off_by_two": off_by_two,
        "off_by_two_rate": off_by_two_rate
    }


# --------------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------------

if __name__ == "__main__":

    # get the cleaned and stratified data from DataPreprocessing
    data = prepare()

    X_train = data["train_projects"]
    X_test = data["test_projects"]

    Y_train = data["y_train"]
    Y_test = data["y_test"]


    # ----------------------------------------------------------
    # TRAINING DATA
    # ----------------------------------------------------------

    print("\nNAIVE BAYES DATA")
    print("============================================")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))


    # ----------------------------------------------------------
    # TRAIN NAIVE BAYES
    # ----------------------------------------------------------

    model = NaiveBayes()

    model.fit(X_train, Y_train)


    # ----------------------------------------------------------
    # MAKE PREDICTIONS
    # ----------------------------------------------------------

    predictions = []

    for record in X_test:

        prediction = model.predict_one(record)

        predictions.append(prediction)

    # ----------------------------------------------------------
    # PRINT ONE SAMPLE EACH
    # ----------------------------------------------------------

    correct_index = 2000 # 3000
    incorrect_index = 2500 # 3010
    off_by_two_index = 194


    # ----------------------------------------------------------
    # CORRECT PREDICTION
    # ----------------------------------------------------------

    print("\n========== CORRECT PREDICTION ==========")

    print("Project:", X_test[correct_index]["description"])
    print("Category:", X_test[correct_index]["category"])
    print("Component Categories:", X_test[correct_index]["componentCategories"])
    print("Region:", X_test[correct_index]["region"])
    print("Province:", X_test[correct_index]["province"])
    print("Contractor:", X_test[correct_index]["contractor"])
    print("Infra Year:", X_test[correct_index]["infraYear"])
    print("Program Name:", X_test[correct_index]["programName"])
    print("Source of Funds:", X_test[correct_index]["sourceOfFunds"])
    print("============================================")
    print("Actual:", Y_test[correct_index])
    print("Predicted:", predictions[correct_index])


    # ----------------------------------------------------------
    # INCORRECT PREDICTION
    # ----------------------------------------------------------

    print("\n========== INCORRECT PREDICTION ==========")

    print("Project:", X_test[incorrect_index]["description"])
    print("Category:", X_test[incorrect_index]["category"])
    print("Component Categories:", X_test[incorrect_index]["componentCategories"])
    print("Region:", X_test[incorrect_index]["region"])
    print("Province:", X_test[incorrect_index]["province"])
    print("Contractor:", X_test[incorrect_index]["contractor"])
    print("Infra Year:", X_test[incorrect_index]["infraYear"])
    print("Program Name:", X_test[incorrect_index]["programName"])
    print("Source of Funds:", X_test[incorrect_index]["sourceOfFunds"])
    print("============================================")
    print("Actual:", Y_test[incorrect_index])
    print("Predicted:", predictions[incorrect_index])


    # ----------------------------------------------------------
    # INCORRECT PREDICTION (OFF BY TWO)
    # ----------------------------------------------------------

    print("\n========== INCORRECT PREDICTION (OFF BY TWO) ==========")

    print("Project:", X_test[off_by_two_index]["description"])
    print("Category:", X_test[off_by_two_index]["category"])
    print("Component Categories:", X_test[off_by_two_index]["componentCategories"])
    print("Region:", X_test[off_by_two_index]["region"])
    print("Province:", X_test[off_by_two_index]["province"])
    print("Contractor:", X_test[off_by_two_index]["contractor"])
    print("Infra Year:", X_test[off_by_two_index]["infraYear"])
    print("Program Name:", X_test[off_by_two_index]["programName"])
    print("Source of Funds:", X_test[off_by_two_index]["sourceOfFunds"])
    print("============================================")
    print("Actual:", Y_test[off_by_two_index])
    print("Predicted:", predictions[off_by_two_index])


    # ----------------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------------

    results = evaluate_model(Y_test, predictions)

    # ----------------------------------------------------------
    # EVALUATION METRICS
    # ----------------------------------------------------------

    print("\nEVALUATION METRICS")
    print("============================================")

    print("Macro-F1 Score:", round(results["macro_f1"], 4))
    print("Accuracy:", round(results["accuracy"], 2), "%")


    # ----------------------------------------------------------
    # OFF-BY-TWO RESULTS PER TIER
    # ----------------------------------------------------------

    print("\nOFF-BY-TWO RATE PER TIER")
    print("============================================")

    for budget_tier in ["Low", "Middle", "High"]:

        print(
            budget_tier + ":",
            results["off_by_two"][budget_tier],
            "errors /",
            round(results["off_by_two_rate"][budget_tier], 2),
            "%"
        )

