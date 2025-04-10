import re
import spacy
from pdfminer.high_level import extract_text

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    return extract_text(pdf_path)

def extract_email(text):
    """Extract email from text using regex."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None

def extract_phone(text):
    """Extract phone number using regex (handles country codes)."""
    phone_pattern = r"\+?\d{1,3}[-.\s]?\d{9,15}"
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else None

def extract_name(text):
    """Extracts name by checking the first two words (fallback to NLP)."""
    lines = text.strip().split("\n")
    if len(lines) > 1:
        potential_name = lines[0].strip()
        if len(potential_name.split()) <= 3:  # Names are usually 2-3 words
            return potential_name

    # Fallback to NLP if the first line doesn't work
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not Found"

def extract_skills(text):
    """Extracts skills using a predefined skill list."""
    skills_list = [
        "Python", "Java", "JavaScript", "C++", "Machine Learning", "AI",
        "NLP", "Flask", "Django", "SQL", "AWS", "Event Management", "MS Office",
        "Problem Solving", "Communication", "Managerial"
    ]
    found_skills = [skill for skill in skills_list if skill.lower() in text.lower()]
    return ", ".join(found_skills) if found_skills else "Not Found"

def extract_education(text):
    """Extracts education details with degree and university name."""
    education_keywords = ["Bachelor", "Master", "PhD", "B.Sc", "M.Sc", "B.Tech", "M.Tech", "Engineering"]
    lines = text.split("\n")
    for line in lines:
        if any(keyword in line for keyword in education_keywords):
            return line.strip()
    return "Not Found"

def extract_experience(text):
    """Extracts experience by looking for work history patterns (dates + job titles)."""
    experience_pattern = r"([A-Za-z]+\s\d{4}\s?-\s?[A-Za-z]*\s?\d{4})"
    experience_matches = re.findall(experience_pattern, text)
    return ", ".join(experience_matches) if experience_matches else "Not Found"

def extract_resume_info(text):
    """Extracts all key information from resume text."""
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "education": extract_education(text),
        "skills": extract_skills(text),
        "experience": extract_experience(text)
    }
