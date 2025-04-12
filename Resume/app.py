from flask import Flask, render_template, request, redirect
import os
import sqlite3
import joblib
import pandas as pd
from src.extractor import extract_text_from_file, extract_resume_info
from recommender import recommend_projects

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ML model and vectorizer
model = joblib.load("models/job_predictor_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Allowed file types
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Save resume data to SQLite
def save_to_db(name, email, phone, education, skills, experience, predicted_job):
    conn = sqlite3.connect("resumes.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            education TEXT,
            skills TEXT,
            experience TEXT,
            predicted_job TEXT
        )
    ''')
    cursor.execute('''
        INSERT INTO resumes (name, email, phone, education, skills, experience, predicted_job)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, email, phone, education, skills, experience, predicted_job))
    conn.commit()
    conn.close()

# Homepage route
@app.route("/")
def index():
    return render_template("index.html")

# File upload handler
@app.route("/upload", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return redirect(request.url)

    file = request.files["resume"]
    if file.filename == "":
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        resume_text = extract_text_from_file(file_path)
        if "Invalid" in resume_text or "Unsupported" in resume_text:
            return render_template("error.html", message=resume_text)

        extracted_info = extract_resume_info(resume_text)

        return render_template(
            "results.html",
            resume_text=resume_text,
            extracted_info=extracted_info
        )

    return render_template("error.html", message="Unsupported file type.")

# Predict job and recommend projects
@app.route("/suggest_job", methods=["POST"])
def suggest_job():
    skills = request.form.get("skills", "")
    name = request.form.get("name", "Unknown")
    email = request.form.get("email", "Unknown")
    phone = request.form.get("phone", "Unknown")
    education= request.form.get("education", "Unknown")
    experience = request.form.get("experience", "Unknown")

    skill_vector = vectorizer.transform([skills])
    predicted_job = model.predict(skill_vector)[0]

    save_to_db(name, email, phone, education, skills, experience, predicted_job)

    # Get recommended projects (always returns up to 3)
    recommended_projects = recommend_projects(predicted_job, skills)

    return render_template("suggestion.html", predicted_job=predicted_job, projects=recommended_projects)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
