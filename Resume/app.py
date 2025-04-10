from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3
from src.extractor import extract_text_from_pdf, extract_resume_info
from job_analyzer import predict_job_role

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Save resume data to database
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

# ✅ Home page
@app.route("/")
def index():
    return render_template("index.html")

# ✅ Upload resume and extract info
@app.route("/upload", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return redirect(request.url)

    file = request.files["resume"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        resume_text = extract_text_from_pdf(file_path)
        extracted_info = extract_resume_info(resume_text)

        return render_template(
            "results.html",
            resume_text=resume_text,
            extracted_info=extracted_info
        )

# ✅ Predict job role from skills and save to DB
@app.route("/suggest_job", methods=["POST"])
def suggest_job():
    skills = request.form.get("skills", "")
    name = request.form.get("name", "Unknown")
    email = request.form.get("email", "Unknown")
    phone = request.form.get("phone", "Unknown")
    education = request.form.get("education", "Unknown")
    experience = request.form.get("experience", "Unknown")

    predicted_job = predict_job_role(skills)
    save_to_db(name, email, phone, education, skills, experience, predicted_job)

    return render_template("suggestion.html", predicted_job=predicted_job)

# ✅ View all resumes stored
# @app.route("/resumes")
# def show_resumes():
#     conn = sqlite3.connect("resumes.db")
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM resumes")
#     resumes = cursor.fetchall()
#     conn.close()
#     return render_template("search.html", resumes=resumes)

# ✅ Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
