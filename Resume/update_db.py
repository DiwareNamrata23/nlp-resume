import sqlite3

conn = sqlite3.connect("resumes.db")
cursor = conn.cursor()

# Add the new column only if it doesn't exist
try:
    cursor.execute("ALTER TABLE resumes ADD COLUMN predicted_job TEXT;")
    print("✅ Column 'predicted_job' added successfully.")
except sqlite3.OperationalError:
    print("⚠️ Column 'predicted_job' might already exist.")

conn.commit()
conn.close()
