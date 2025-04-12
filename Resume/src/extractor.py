import os
import re
import spacy
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from docx import Document

# Load spaCy model globally
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("spaCy model loading failed:", e)
    nlp = None

def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except PDFSyntaxError:
        return "Invalid or corrupted PDF file."

def extract_text_from_docx(path):
    try:
        doc = Document(path)
        full_text = "\n".join([para.text for para in doc.paragraphs])
        return full_text
    except Exception as e:
        return f"Invalid DOCX file: {str(e)}"

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        return "Unsupported file format"

def extract_email(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not Found"

def extract_phone(text):
    phone_pattern = r"\+?\d{1,3}[-.\s]?\d{9,15}"
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else "Not Found"

def extract_name(text):
    lines = text.strip().split("\n")
    if len(lines) > 1:
        potential_name = lines[0].strip()
        if len(potential_name.split()) <= 3:
            return potential_name

    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text

    return "Not Found"

def extract_skills(text):
    skills_list = [
        "Python", "Java", "JavaScript", "C++", "Machine Learning", "AI",
        "NLP", "Flask", "Django", "SQL", "AWS", "Event Management", "MS Office",
        "Problem Solving", "Communication", "Managerial"
    ]
    found_skills = [skill for skill in skills_list if skill.lower() in text.lower()]
    return ", ".join(found_skills) if found_skills else "Not Found"

def extract_education(text):
    education_keywords = ["Bachelor", "Master", "PhD", "B.Sc", "M.Sc", "B.Tech", "M.Tech", "Engineering"]
    lines = text.split("\n")
    for line in lines:
        if any(keyword in line for keyword in education_keywords):
            return line.strip()
    return "Not Found"

def extract_experience(text):
    experience_pattern = r"([A-Za-z]+\s\d{4}\s?-\s?[A-Za-z]*\s?\d{4})"
    experience_matches = re.findall(experience_pattern, text)
    return ", ".join(experience_matches) if experience_matches else "Not Found"

def extract_resume_info(text):
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "education": extract_education(text),
        "skills": extract_skills(text),
        "experience": extract_experience(text)
    }
