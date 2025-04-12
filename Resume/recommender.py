import pandas as pd
import random

# Load and clean the dataset
try:
    project_df = pd.read_csv("job_projects_dataset.csv", usecols=["Job Role", "Skill", "Project"])
except Exception as e:
    print("Error loading project dataset:", e)
    project_df = pd.DataFrame(columns=["Job Role", "Skill", "Project"])

def recommend_projects(predicted_job, skills):
    # Ensure skills is a string and split it
    if isinstance(skills, str):
        skills = [skill.strip().lower() for skill in skills.split(",")]

    # Filter the dataset
    relevant_projects = project_df[
        project_df["Job Role"].str.lower() == predicted_job.lower()
    ]

    recommended = []
    for skill in skills:
        matches = relevant_projects[relevant_projects["Skill"].str.lower() == skill]
        for _, row in matches.iterrows():
            if row["Project"] not in recommended:
                recommended.append(row["Project"])

    # If fewer than 3, randomly sample more
    if len(recommended) < 3:
        extra = list(set(relevant_projects["Project"]) - set(recommended))
        recommended.extend(random.sample(extra, min(3 - len(recommended), len(extra))))

    return recommended
