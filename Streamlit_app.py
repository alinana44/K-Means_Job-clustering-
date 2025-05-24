import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
from job_utils import (
    scrape_karkidi_jobs,
    preprocess_job_data,
    train_clustering_model,
    load_latest_model,
    classify_single_job,
    classify_jobs_from_csv,
    analyze_classification_results,
    load_latest_jobs_csv
)

st.set_page_config(page_title="Job Clustering & Classification", layout="wide")
st.title("🔍 Job Posting Classifier & Clusterer")

# Sidebar
with st.sidebar:
    st.header("📁 Navigation")
    selection = st.radio("Choose a Task", [
        "Scrape Job Listings",
        "Train Clustering Model",
        "Classify a Job",
        "Batch Classify from CSV"
    ])

# Page 1: Scrape Job Listings
if selection == "Scrape Job Listings":
    st.header("📥 Scrape Job Listings")
    keyword = st.text_input("Enter search keyword:", value="data science")
    pages = st.slider("Number of pages to scrape", min_value=1, max_value=10, value=2)
    if st.button("Scrape Jobs"):
        df = scrape_karkidi_jobs(keyword, pages)
        if not df.empty:
            st.success(f"✅ Scraped {len(df)} jobs.")
            filename = f"jobs_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            st.download_button("📥 Download CSV", df.to_csv(index=False), file_name=filename)
            st.dataframe(df.head())
        else:
            st.warning("⚠️ No jobs found. Try different keywords or check your connection.")

# Page 2: Train Clustering Model
elif selection == "Train Clustering Model":
    st.header("📊 Train Clustering Model")
    if st.button("Load Latest Jobs and Train Model"):
        df = load_latest_jobs_csv()
        if df is not None and len(df) >= 4:
            df = preprocess_job_data(df)
            df_clustered, model_data, model_filename = train_clustering_model(df)
            clustered_filename = f"jobs_clustered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_clustered.to_csv(clustered_filename, index=False)
            st.success(f"✅ Model saved as: {model_filename}")
            st.success(f"✅ Clustered data saved as: {clustered_filename}")
            st.dataframe(df_clustered.head())
        elif df is not None:
            st.warning("⚠️ Not enough jobs (minimum 4) to perform clustering.")
        else:
            st.warning("⚠️ No jobs CSV file found. Please scrape jobs first.")

# Page 3: Classify a Single Job
elif selection == "Classify a Job":
    st.header("🧠 Classify a Job")
    model_data = load_latest_model()
    if model_data:
        title = st.text_input("Job Title")
        company = st.text_input("Company")
        location = st.text_input("Location")
        skills = st.text_input("Skills (comma-separated)")
        summary = st.text_area("Job Summary")

        if st.button("Classify Job"):
            job_data = {
                'title': title,
                'company': company,
                'location': location,
                'skills': skills,
                'summary': summary
            }
            result = classify_single_job(job_data, model_data)
            if result:
                st.success(f"🔖 Cluster ID: {result['cluster_id']}")
                st.info(f"📈 Confidence Score: {result['confidence']:.3f}")
                st.code(result['processed_skills'], language='text')
            else:
                st.error("❌ Failed to classify the job.")
    else:
        st.warning("⚠️ No model found. Please train a model first.")

# Page 4: Batch Classification
elif selection == "Batch Classify from CSV":
    st.header("📦 Batch Classify Jobs from CSV")
    model_data = load_latest_model()
    if model_data:
        uploaded_file = st.file_uploader("Upload a CSV with job postings", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.info(f"📄 Loaded {len(df)} jobs for classification.")
            results = classify_jobs_from_csv(uploaded_file.name, model_data)
            if not results.empty:
                st.dataframe(results.head())
                output_filename = f"classified_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results.to_csv(output_filename, index=False)
                st.download_button("📥 Download Results", results.to_csv(index=False), file_name=output_filename)
                analyze_classification_results(results)
    else:
        st.warning("⚠️ No model found. Please train a model first.")
