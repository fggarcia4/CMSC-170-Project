# IMPORTS
#------------------------------------------
import csv
import os
import sys
import ast
import random
import numpy
#------------------------------------------

#------------------------------------------
CSV_NAME = "dpwh_transparency_data.csv"

TIERS = ["Low", "Middle", "High"]
LOW_MAX = 5_000_000
MIDDLE_MAX = 18_999_999

INPUT_COLUMNS = [
    "category",
    "componentCategories",
    "location",
    "contractor",
    "infraYear",
    "programName",
    "sourceOfFunds"
]
#------------------------------------------

# load csv file
#------------------------------------------
def load_csv():
    with open(CSV_NAME, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    return rows
#------------------------------------------

# filter years from 2023 to 2025 only
#------------------------------------------
def filter_years(rows):
    filtered_rows = []

    for row in rows:
        year = row["infraYear"].strip()

        if year in ["2023", "2024", "2025"]:
            filtered_rows.append(row)

    return filtered_rows
#------------------------------------------

# clean category
#------------------------------------------
def clean_category(value):
    if value is None or value.strip() == "":
        return None

    return value.strip()
#------------------------------------------

# clean componentCategories
#------------------------------------------
def clean_component_categories(value):
    if value is None or value.strip() == "":
        return None
    return value.strip()
#------------------------------------------

# clean location
#------------------------------------------
def clean_location(value):
    try:
        location = ast.literal_eval(value)

        region = location["region"].strip()
        province = location["province"].strip()

        if not region or not province:
            return None, None

        return region, province

    except(ValueError, SyntaxError, KeyError, TypeError):
        return None, None
#------------------------------------------

# clean contractor
#------------------------------------------
def clean_contractor(value):
    if value is None or value.strip() == "":
        return None
    return value.strip()
#------------------------------------------

# clean infraYear
#------------------------------------------
def clean_infra_year(value):
    try:
        year = int(value.strip())

        if year not in [2023, 2024, 2025]:
            return None

        return year
    except(ValueError, AttributeError):
        return None
#------------------------------------------

# clean programName
#------------------------------------------
def clean_program_name(value):
    if value is None or value.strip() == "":
        return None

    return value.strip()
#------------------------------------------

# clean sourceOfFunds
#------------------------------------------
def clean_source_of_funds(value):
    if value is None or value.strip() == "":
        return None

    return value.strip()
#------------------------------------------

# clean budget
#------------------------------------------
def clean_budget(value):
    try:
        budget = float(value.strip())
        if budget <= 0:
            return None
        return budget
    except(ValueError, AttributeError):
        return None
#------------------------------------------

# convert budget to tier
#------------------------------------------
def budget_to_tier(budget):
    if budget <= LOW_MAX:
        return "Low"
    elif budget <= MIDDLE_MAX:
        return "Middle"
    else:
        return "High"
#------------------------------------------

# clean all projects
#------------------------------------------
def clean_projects(rows):
    cleaned_rows = []

    for row in rows:
        category = clean_category(row["category"])
        componentCategories = clean_component_categories(row["componentCategories"])
        region, province = clean_location(row["location"])
        contractor = clean_contractor(row["contractor"])
        infraYear = clean_infra_year(row["infraYear"])
        programName = clean_program_name(row["programName"])
        sourceOfFunds = clean_sourch_of_funds(row["sourceOfFunds"])
        budget = clean_budget(row["budget"])

        if category is None:
            continue
        if componentCategories is None:
            continue
        if region is None or province is None:
            continue
        if contractor is None:
            continue
        if infraYear is None:
            continue
        if programName is None:
            continue
        if sourceOfFunds is None:
            continue
        if budget is None:
            continue

        row["category"] = category
        row["componentCategories"] = componentCategories
        row["region"] = region
        row["province"] = province
        row["contractor"] = contractor
        row["infraYear"] = infraYear
        row["programName"] = programName
        row["sourceOfFunds"] = sourceOfFunds
        row["budget"] = budget
        row["tier"] = budget_to_tier(budget)

        cleaned_rows.append(row)
    return cleaned_rows
#------------------------------------------

# get tiers
#------------------------------------------
def get_tiers(projects):
    tiers = []

    for row in projects:
        tiers.append(row["tier"])

    return np.array(tiers)
#------------------------------------------

# split data into training and testing
#------------------------------------------
def stratified_split(tiers, test_frac=0.2, seed=42):
    random.seed(seed)
    by_tier = {}

    for i, tiers in enumerate(tiers):
        if tiers not in by_tier:
            by_tier[tiers] = []
        by_tier[tiers].append(i)
    
    train = []
    test = []

    for tiers in by_tier:
        indexes = by_tier[tiers][:]
        random.shuffle(indexes)
        test_count = round(len(indexes) * test_frac)
        test.extend(indexes[:test_count])
        train.extend(indexes[test_count:])
    random.shuffle(train)
    random.shuffle(test)

    return np.array(train), np.array(test)
#------------------------------------------

# get projects using indexes
#------------------------------------------
def subset(projects, indexes):
    selected =[]
    for i in indexes:
        selected.append(projects[i])
    return selected
#------------------------------------------

# prepare the dataset
#------------------------------------------
def prepare(path=None, test_frac=0.2, seed=42):
    rows = load_csv()
    rows = filter_years(rows)
    projects = clean_projects(rows)
    tiers = get_tiers(projects)

    train, test = stratified_split(
        tiers,
        test_frac,
        seed
    )

    train_projects = subset(projects, train)
    test_projects = subset(projects, test)

    return {
        "projects": projects,
        "train_projects": train_projects,
        "test_projects": test_projects,
        "y_train": tiers[train],
        "y_test": tiers[test],
        "train": train,
        "test": test
    }
#------------------------------------------

# main program
#------------------------------------------
if __name__ == "__main__":
    data = prepare()
#------------------------------------------
