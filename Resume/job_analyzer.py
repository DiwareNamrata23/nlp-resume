import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def predict_job_role(skills):
    try:
        # Load job descriptions
        df = pd.read_csv("job_descriptions.csv")

        # Combine the skills from resume with all job descriptions
        corpus = df['description'].tolist()
        corpus.append(skills.lower())

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Compute cosine similarity
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        # Find best match
        best_match_idx = similarities.argmax()
        best_job = df.iloc[best_match_idx]['job_title']
        return best_job if similarities[best_match_idx] > 0 else "Other"
    except Exception as e:
        print("Error in job prediction:", e)
        return "Unknown"
