import csv
import random


dataset = "dpwh_transparency_data.csv"

# DATA CLEANING 
# ---------------------------------------------------------------------------------
def get_region(location):
    if "'region': '" in location:
        return location.split("'region': '")[1].split("'")[0]

    return ""

def get_budget_tier(budget):
    budget = float(budget)

    if budget <= 0:
        return None
    
    elif budget <= 5000000:
        return "Low"

    elif budget <= 8999999:
        return "Middle"

    else:
        return "High"

def prepare_data():
    projects = []

    with open(dataset, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            category = row["componentCategories"].strip()

            if not category:
                continue

            region = get_region(row["location"])

            year = int(row["infraYear"])

            budget_tier = get_budget_tier(row["budget"])

            if budget_tier is None:
                continue
            
            project = {
                "category": category,
                "region": region,
                "year": year,
                "budget_tier": budget_tier
            }

            projects.append(project)

    return projects
        

def load_dataset():
    with open(dataset, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        total_records = 0
        missing_category = 0
        zero_budget = 0

        low = 0
        middle = 0
        high = 0

        for row in reader:
            total_records += 1

            if not row["componentCategories"].strip():
                continue
                missing_category += 1

            budget_tier = get_budget_tier(row["budget"])

            if budget_tier is None:
                zero_budget += 1

            elif budget_tier == "Low":
                low += 1

            elif budget_tier == "Middle":
                middle += 1

            elif budget_tier == "High":
                high += 1


        print("\nBUDGET CLASSIFICATION")
        print("-------------------------")
        print("Total Records:", total_records)
        print("Missing Categories:", missing_category)
        print("Zero/Negative Budgets:", zero_budget)
        print("Low:", low)
        print("Middle:", middle)
        print("High:", high)

projects = prepare_data()

inputs = []
target = []

for project in projects:
    inputs.append([
        project["category"],
        project["region"],
        project["year"]
    ])

    target.append(project["budget_tier"])

print("\nInput: ")
print(inputs[:10])

print("\nTarget: ")
print(target[:10])
# ---------------------------------------------------------------------------------

