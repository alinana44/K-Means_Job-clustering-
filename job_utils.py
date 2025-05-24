# Step 1: Job Scraper
# This script scrapes jobs from karkidi.com and saves them to CSV

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

def scrape_karkidi_jobs(keyword="data science", pages=2):
    """Scrape jobs from karkidi.com"""
    print(f"Scraping jobs for keyword: '{keyword}'")
    print("=" * 50)


    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        try:
            url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
            print(f"Scraping page: {page}")
            print(f"URL: {url}")

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            job_blocks = soup.find_all("div", class_="ads-details")

            print(f"Found {len(job_blocks)} job blocks on page {page}")

            for i, job in enumerate(job_blocks):
                try:
                    # Extract job details
                    title = job.find("h4")
                    title = title.get_text(strip=True) if title else "N/A"

                    company_element = job.find("a", href=lambda x: x and "Employer-Profile" in x)
                    company = company_element.get_text(strip=True) if company_element else "N/A"

                    location_element = job.find("p")
                    location = location_element.get_text(strip=True) if location_element else "N/A"

                    experience_element = job.find("p", class_="emp-exp")
                    experience = experience_element.get_text(strip=True) if experience_element else "N/A"

                    # Extract skills
                    key_skills_tag = job.find("span", string="Key Skills")
                    skills = ""
                    if key_skills_tag:
                        skills_p = key_skills_tag.find_next("p")
                        skills = skills_p.get_text(strip=True) if skills_p else ""

                    # Extract summary
                    summary_tag = job.find("span", string="Summary")
                    summary = ""
                    if summary_tag:
                        summary_p = summary_tag.find_next("p")
                        summary = summary_p.get_text(strip=True) if summary_p else ""

                    # Create unique job ID
                    job_id = f"{company}_{title}_{location}".replace(" ", "_").lower()
                    job_id = re.sub(r'[^a-zA-Z0-9_]', '', job_id)[:50]

                    job_data = {
                        "job_id": job_id,
                        "title": title,
                        "company": company,
                        "location": location,
                        "experience": experience,
                        "skills": skills,
                        "summary": summary,
                        "scraped_date": datetime.now().isoformat()
                    }

                    jobs_list.append(job_data)
                    print(f"  ✓ Job {i+1}: {title} at {company}")

                except Exception as e:
                    print(f"  ✗ Error parsing job {i+1}: {e}")
                    continue

            time.sleep(1)  # Be nice to the server

        except Exception as e:
            print(f"Error scraping page {page}: {e}")
            continue

    print(f"\nTotal jobs scraped: {len(jobs_list)}")
    return pd.DataFrame(jobs_list)

if __name__ == "__main__":
    # Get user input
    keyword = input("Enter search keyword (default: 'data science'): ").strip()
    if not keyword:
        keyword = "data science"

    pages_input = input("Enter number of pages to scrape (default: 2): ").strip()
    try:
        pages = int(pages_input) if pages_input else 2
    except ValueError:
        pages = 2

    # Scrape jobs
    df_jobs = scrape_karkidi_jobs(keyword=keyword, pages=pages)

    if len(df_jobs) > 0:
        # Save to CSV
        filename = f"jobs_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_jobs.to_csv(filename, index=False)
        print(f"\n✓ Jobs saved to: {filename}")

        # Display sample
        print(f"\nFirst 3 jobs:")

        print("=" * 80)
        for idx, row in df_jobs.head(3).iterrows():
            print(f"Title: {row['title']}")
            print(f"Company: {row['company']}")
            print(f"Location: {row['location']}")
            print(f"Skills: {row['skills']}")
            print("-" * 40)
    else:
        print("No jobs found. Check your internet connection or try different keywords.")

"""# K-Means Clustering"""

# Step 2: Job Clustering with K-means
# This script loads the scraped jobs and creates clusters using K-means

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import os
import glob
from datetime import datetime

def extract_skills_from_text(text):
    """Extract technical skills from job text using keyword matching"""
    # Common technical skills - you can expand this list
    skill_keywords = [
        'python', 'java', 'javascript', 'react', 'nodejs', 'angular', 'vue',
        'php', 'laravel', 'django', 'flask', 'sql', 'mysql', 'postgresql',
        'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
        'git', 'linux', 'html', 'css', 'bootstrap', 'jquery', 'typescript',
        'c++', 'c#', 'ruby', 'rails', 'go', 'rust', 'scala', 'kotlin',
        'swift', 'flutter', 'react native', 'android', 'ios', 'unity',
        'tensorflow', 'pytorch', 'machine learning', 'data science',
        'artificial intelligence', 'blockchain', 'devops', 'ci/cd',
        'selenium', 'jenkins', 'ansible', 'terraform', 'microservices',
        'api', 'rest', 'graphql', 'agile', 'scrum', 'jira'
    ]

    if pd.isna(text) or text == "":
        return 'general'

    text_lower = text.lower()
    found_skills = []

    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill)

    return ', '.join(found_skills) if found_skills else 'general'

def preprocess_job_data(df):
    """Preprocess job data for clustering"""
    print("Preprocessing job data...")

    # Handle missing skills by extracting from title and summary
    df['processed_skills'] = df.apply(lambda row:
        row['skills'] if (pd.notna(row['skills']) and row['skills'].strip())
        else extract_skills_from_text(str(row['title']) + " " + str(row['summary'])),
        axis=1
    )

    print(f"Skills extracted for {len(df)} jobs")
    return df

def find_optimal_clusters(X, max_k=8):
    """Find optimal number of clusters using silhouette score"""
    print("Finding optimal number of clusters...")

    if X.shape[0] < 4:
        print("Not enough data points for clustering. Using 2 clusters.")
        return 2

    max_k = min(max_k, X.shape[0] - 1)
    k_range = range(2, max_k + 1)

    silhouette_scores = []
    inertias = []

    for k in k_range:
        print(f"  Testing {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans.inertia_)

        print(f"    Silhouette Score: {silhouette_avg:.3f}")

    # Find optimal k using silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    print(f"\n✓ Optimal number of clusters: {optimal_k}")
    print(f"✓ Best silhouette score: {best_score:.3f}")

    return optimal_k

def train_clustering_model(df):
    """Train K-means clustering model"""
    print("Training clustering model...")
    print("=" * 50)

    # Create combined text for clustering (skills + job title)
    df['combined_text'] = df['processed_skills'] + ' ' + df['title'].fillna('')

    # Create TF-IDF vectors
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,  # Minimum document frequency
        max_df=0.95  # Maximum document frequency
    )

    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Scale the features
    scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
    tfidf_scaled = scaler.fit_transform(tfidf_matrix)

    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(tfidf_scaled)

    # Train final model
    print(f"\nTraining final K-means model with {optimal_k} clusters...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_scaled)

    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels

    # Save the model and components
    model_data = {
        'kmeans_model': kmeans,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'feature_names': vectorizer.get_feature_names_out()
    }

    model_filename = f"job_clustering_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(model_data, model_filename)
    print(f"✓ Model saved as: {model_filename}")

    return df, model_data, model_filename

def analyze_clusters(df):
    """Analyze and display cluster characteristics"""
    print("\nCluster Analysis:")
    print("=" * 60)

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_jobs = df[df['cluster'] == cluster_id]

        print(f"\n🔹 Cluster {cluster_id} ({len(cluster_jobs)} jobs)")
        print("-" * 40)

        # Most common skills in this cluster
        all_skills_text = ' '.join(cluster_jobs['processed_skills'].fillna(''))
        if all_skills_text.strip():
            skills_list = [skill.strip() for skill in all_skills_text.split(',') if skill.strip()]
            if skills_list:
                skill_counts = pd.Series(skills_list).value_counts().head(5)
                print("Top skills:")
                for skill, count in skill_counts.items():
                    percentage = (count / len(cluster_jobs)) * 100
                    print(f"  • {skill}: {count} jobs ({percentage:.1f}%)")

        # Sample job titles
        print("\nSample job titles:")
        sample_titles = cluster_jobs['title'].head(3).tolist()
        for i, title in enumerate(sample_titles, 1):
            print(f"  {i}. {title}")

        # Most common companies
        if len(cluster_jobs) > 1:
            company_counts = cluster_jobs['company'].value_counts().head(3)
            print("\nTop companies:")
            for company, count in company_counts.items():
                print(f"  • {company}: {count} jobs")

def load_latest_jobs_csv():
    """Load the most recent jobs CSV file"""
    csv_files = glob.glob("jobs_*.csv")
    if not csv_files:
        print("No jobs CSV files found. Please run Step 1 first.")
        return None

    # Get the most recent file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"Loading jobs from: {latest_file}")

    df = pd.read_csv(latest_file)
    print(f"Loaded {len(df)} jobs")
    return df

if __name__ == "__main__":
    print("Job Clustering with K-means")
    print("=" * 40)

    # Load jobs data
    df = load_latest_jobs_csv()
    if df is None:
        exit(1)

    # Check if we have enough data
    if len(df) < 4:
        print("Not enough jobs for meaningful clustering. Please scrape more jobs.")
        exit(1)

    # Preprocess data
    df = preprocess_job_data(df)

    # Train clustering model
    df_clustered, model_data, model_filename = train_clustering_model(df)

    # Save clustered data
    clustered_filename = f"jobs_clustered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_clustered.to_csv(clustered_filename, index=False)
    print(f"✓ Clustered jobs saved as: {clustered_filename}")

    # Analyze clusters
    analyze_clusters(df_clustered)

    print(f"\n" + "=" * 60)
    print(" Clustering completed successfully!")
    print(f" Model saved as: {model_filename}")
    print(f" Clustered data saved as: {clustered_filename}")
    print("\nNext: Run Step 3 to classify new jobs using this model.")

"""# classifying new jobs using the trained model"""

# Step 3: New Job Classifier
# This script classifies new jobs using the trained model

import pandas as pd
import joblib
import glob
import os
from datetime import datetime

def load_latest_model():
    """Load the most recent clustering model"""
    model_files = glob.glob("job_clustering_model_*.pkl")
    if not model_files:
        print(" No trained model found. Please run Step 2 first.")
        return None

    # Get the most recent model file
    latest_model = max(model_files, key=os.path.getctime)
    print(f"📂 Loading model from: {latest_model}")

    try:
        model_data = joblib.load(latest_model)
        print("✅ Model loaded successfully!")
        return model_data
    except Exception as e:
        print(f" Error loading model: {e}")
        return None

def extract_skills_from_text(text):
    """Extract technical skills from job text using keyword matching"""
    skill_keywords = [
        'python', 'java', 'javascript', 'react', 'nodejs', 'angular', 'vue',
        'php', 'laravel', 'django', 'flask', 'sql', 'mysql', 'postgresql',
        'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
        'git', 'linux', 'html', 'css', 'bootstrap', 'jquery', 'typescript',
        'c++', 'c#', 'ruby', 'rails', 'go', 'rust', 'scala', 'kotlin',
        'swift', 'flutter', 'react native', 'android', 'ios', 'unity',
        'tensorflow', 'pytorch', 'machine learning', 'data science',
        'artificial intelligence', 'blockchain', 'devops', 'ci/cd',
        'selenium', 'jenkins', 'ansible', 'terraform', 'microservices',
        'api', 'rest', 'graphql', 'agile', 'scrum', 'jira'
    ]

    if pd.isna(text) or text == "":
        return 'general'

    text_lower = text.lower()
    found_skills = []

    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill)

    return ', '.join(found_skills) if found_skills else 'general'

def classify_single_job(job_data, model_data):
    """Classify a single job using the trained model"""
    try:
        # Extract components from model
        kmeans = model_data['kmeans_model']
        vectorizer = model_data['vectorizer']
        scaler = model_data['scaler']

        # Process job skills
        skills = job_data.get('skills', '')
        if not skills or pd.isna(skills):
            skills = extract_skills_from_text(str(job_data.get('title', '')) + " " + str(job_data.get('summary', '')))

        # Create combined text
        combined_text = skills + ' ' + str(job_data.get('title', ''))

        # Transform using the same pipeline as training
        tfidf_vector = vectorizer.transform([combined_text])
        tfidf_scaled = scaler.transform(tfidf_vector)

        # Predict cluster
        cluster_id = kmeans.predict(tfidf_scaled)[0]

        # Get prediction confidence (distance to cluster center)
        distances = kmeans.transform(tfidf_scaled)[0]
        confidence = 1 / (1 + distances[cluster_id])  # Convert distance to confidence

        return {
            'cluster_id': int(cluster_id),
            'confidence': float(confidence),
            'processed_skills': skills
        }

    except Exception as e:
        print(f" Error classifying job: {e}")
        return None

def classify_jobs_from_csv(csv_file, model_data):
    """Classify all jobs from a CSV file"""
    print(f"--->> Classifying jobs from: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} jobs to classify")

    results = []

    for idx, row in df.iterrows():
        job_data = row.to_dict()
        classification = classify_single_job(job_data, model_data)

        if classification:
            # Add classification results to the job data
            job_data.update(classification)
            results.append(job_data)

            print(f"✓ Job {idx+1}: '{row['title']}' → Cluster {classification['cluster_id']} (confidence: {classification['confidence']:.3f})")
        else:
            print(f" Failed to classify job {idx+1}: '{row['title']}'")

    return pd.DataFrame(results)

def analyze_classification_results(df):
    """Analyze and display classification results"""
    print(f"\n--->> Classification Results Summary:")
    print("=" * 50)

    # Cluster distribution
    cluster_counts = df['cluster_id'].value_counts().sort_index()
    print(f"\n--->> Jobs per cluster:")
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Cluster {cluster_id}: {count} jobs ({percentage:.1f}%)")

    # Average confidence per cluster
    print(f"\n-->> Average confidence per cluster:")
    confidence_by_cluster = df.groupby('cluster_id')['confidence'].mean()
    for cluster_id, avg_conf in confidence_by_cluster.items():
        print(f"  Cluster {cluster_id}: {avg_conf:.3f}")

    # Show sample jobs from each cluster
    print(f"\n-->> Sample jobs from each cluster:")
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_jobs = df[df['cluster_id'] == cluster_id]
        print(f"\n  🔹 Cluster {cluster_id} sample:")

        # Show top 2 jobs with highest confidence
        top_jobs = cluster_jobs.nlargest(2, 'confidence')
        for idx, (_, job) in enumerate(top_jobs.iterrows(), 1):
            print(f"    {idx}. {job['title']} at {job['company']}")
            print(f"       Skills: {job['processed_skills']}")
            print(f"       Confidence: {job['confidence']:.3f}")

def classify_manual_job():
    """Allow user to manually input a job and classify it"""
    print(f"\n-->> Manual Job Classification")
    print("=" * 40)

    job_data = {}
    job_data['title'] = input("Enter job title: ").strip()
    job_data['company'] = input("Enter company name: ").strip()
    job_data['location'] = input("Enter location: ").strip()
    job_data['skills'] = input("Enter skills (comma-separated): ").strip()
    job_data['summary'] = input("Enter job summary/description: ").strip()

    return job_data

if __name__ == "__main__":
    print("New Job Classifier")
    print("=" * 30)

    # Load the trained model
    model_data = load_latest_model()
    if model_data is None:
        exit(1)

    while True:
        print(f"\nOptions:")
        print("1. Classify jobs from CSV file")
        print("2. Classify a single job manually")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            # Find available CSV files
            csv_files = glob.glob("jobs_*.csv")
            csv_files = [f for f in csv_files if not f.startswith("jobs_clustered_")]

            if not csv_files:
                print(" No job CSV files found. Please run Step 1 first.")
                continue

            print(f"\nAvailable CSV files:")
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {file}")

            try:
                file_choice = int(input("Enter file number: ")) - 1
                if 0 <= file_choice < len(csv_files):
                    selected_file = csv_files[file_choice]

                    # Classify jobs
                    classified_df = classify_jobs_from_csv(selected_file, model_data)

                    if len(classified_df) > 0:
                        # Save results
                        output_filename = f"jobs_classified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        classified_df.to_csv(output_filename, index=False)
                        print(f"\n--->> Classification results saved as: {output_filename}")

                        # Analyze results
                        analyze_classification_results(classified_df)
                else:
                    print("Invalid file number.")
            except ValueError:
                print("Please enter a valid number.")

        elif choice == '2':
            # Manual job classification
            job_data = classify_manual_job()

            classification = classify_single_job(job_data, model_data)

            if classification:
                print(f"\n--->> Classification Result:")
                print(f"📂 Cluster ID: {classification['cluster_id']}")
                print(f"-->> Confidence: {classification['confidence']:.3f}")
                print(f"-->> Processed Skills: {classification['processed_skills']}")

                # Save single job result
                result_data = job_data.copy()
                result_data.update(classification)
                result_df = pd.DataFrame([result_data])

                output_filename = f"single_job_classified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                result_df.to_csv(output_filename, index=False)
                print(f"--->> Result saved as: {output_filename}")
            else:
                print("-->> Failed to classify the job.")

        elif choice == '3':
            print(" Exited")
            break

        else:
            print("Invalid choice. Please try again.")

"""# Alerting System"""

import csv

# Data
header = ['user_id', 'email', 'preferred_keywords']
rows = [
    [1, 'alice@example.com', 'machine learning, data science, AI'],
    [2, 'bob@example.com', 'web development, react, frontend'],
    [3, 'alina.ds24@duk.ac.in', 'python, programming, coding'],
]

# Create CSV
with open('user_preferences.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("CSV file 'users_preferences.csv' created successfully.")



import pandas as pd
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yagmail

# Load user preferences
def load_user_preferences(path='user_preferences.csv'):
    return pd.read_csv(path)

# Load the specified or latest clustered job file
def load_new_jobs(path):
    return pd.read_csv(path)

# Send an email notification using yagmail
def send_email_alert(user_email, job, sender_email, sender_password):
    yag = yagmail.SMTP(sender_email, sender_password)

    subject = f"New Job Match: {job['title']} at {job['company']}"
    body = f"""
 **Job Title:** {job['title']}
 **Company:** {job['company']}
 **Skills:** {job['processed_skills']}
 **Summary:** {job.get('summary', 'N/A')}
 **Link:** {job.get('link', 'N/A')}

This job matches your preferences!
"""

    try:
        yag.send(to=user_email, subject=subject, contents=body)
        print(f" Email sent to {user_email}")
    except Exception as e:
        print(f" Failed to send email to {user_email}. Error: {e}")

# Match jobs to each user's preferred keywords
def match_jobs_to_users(users, jobs, sender_email, sender_password):
    vectorizer = TfidfVectorizer()

    for _, user in users.iterrows():
        preferred_keywords = user['preferred_keywords']

        for _, job in jobs.iterrows():
            job_summary = job.get('summary', '') or ''
            job_skills = job.get('processed_skills', '') or ''

            job_text = f"{job.get('title', '')} {job_skills} {job_summary}"

            combined_corpus = [preferred_keywords, job_text]
            try:
                if preferred_keywords.strip() == "" or job_text.strip() == "":
                    similarity = 0.0
                else:
                    combined_tfidf = vectorizer.fit_transform(combined_corpus)
                    similarity = cosine_similarity(combined_tfidf[0:1], combined_tfidf[1:2])[0][0]
            except ValueError as e:
                print(f"⚠️ Could not compute similarity for user '{user['email']}' and job '{job.get('title', 'N/A')}'. Error: {e}")
                similarity = 0.0

            if similarity >= 0.1:
                send_email_alert(user['email'], job, sender_email, sender_password)

# Main function
if __name__ == "__main__":
    import getpass

    print(" Starting Job Matching Alert System...")

    # Ask for Gmail credentials securely
    sender_email = input("Enter your Gmail address: ")
    sender_password = getpass.getpass("Enter your Gmail App Password (not your normal password): ")

    # Load users
    users_df = load_user_preferences()

    # Locate the latest clustered jobs file
    clustered_files = glob.glob("jobs_clustered_*.csv")
    if not clustered_files:
        print(" No clustered job files found. Please run Step 2 (job clustering).")
        jobs_df = pd.DataFrame()
    else:
        latest_clustered_file = max(clustered_files, key=os.path.getctime)
        print(f" Loading job data from: {latest_clustered_file}")
        jobs_df = load_new_jobs(latest_clustered_file)

    if not users_df.empty and not jobs_df.empty:
        match_jobs_to_users(users_df, jobs_df, sender_email, sender_password)
    elif users_df.empty:
        print(" No user preferences found. Please check 'user_preferences.csv'.")
    elif jobs_df.empty:
        print(" No job data found.")

    print(" Job Matching Finished.")
