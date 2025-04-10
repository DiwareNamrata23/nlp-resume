import sqlite3

def create_database():
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
        experience TEXT
    )
    ''')

    conn.commit()
    conn.close()

# Execute database creation
create_database()
print("Database created successfully!")
