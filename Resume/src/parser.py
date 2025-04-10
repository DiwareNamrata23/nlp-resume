import os
import pdfminer.high_level
import docx

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = pdfminer.high_level.extract_text(pdf_path)
    return text.strip()

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_resume_text(file_path):
    """Extract text from a resume (PDF or DOCX)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

# Test with a sample resume
# if __name__ == "__main__":
#     sample_resume = "sample_resume.pdf"  # Change to your file path
#     text = extract_resume_text(sample_resume)
#     print(text[:500])  # Print first 500 characters to verify extraction
