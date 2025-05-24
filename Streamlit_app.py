# app.py
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

tab1, tab2, tab3 = st.tabs(["📥 Scrape Jobs", "📊 Train & Cluster", "🧠 Classify Jobs"])

with tab1:
    st.subheader("Step 1: Scrape Job Listings")
    keyword = st.text_input("Enter search keyword:", value="data science")
    pages = st.slider("Number of pages to scrape", min_value=1, max_value=10, value=2)
    if st.button("Scrape Now"):
        df = scrape_karkidi_jobs(keyword, pages)
        st.success(f"Scraped {len(df)} jobs.")
        filename = f"jobs_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=filename)

with tab2:
    st.subheader("Step 2: Train Clustering Model")
    if st.button("Load Latest Jobs & Train"):
        df = load_latest_jobs_csv()
        if df is not None:
            df = preprocess_job_data(df)
            df_clustered, model_data, model_filename = train_clustering_model(df)
            st.success(f"Model saved as: {model_filename}")
            st.dataframe(df_clustered.head())
        else:
            st.warning("No jobs CSV found. Please scrape first.")

with tab3:
    st.subheader("Step 3: Classify a Job")
    model_data = load_latest_model()
    if model_data:
        title = st.text_input("Job Title")
        company = st.text_input("Company")
        location = st.text_input("Location")
        skills = st.text_input("Skills (comma-separated)")
        summary = st.text_area("Job Summary")

        if st.button("Classify"):
            job_data = {
                'title': title,
                'company': company,
                'location': location,
                'skills': skills,
                'summary': summary
            }
            result = classify_single_job(job_data, model_data)
            if result:
                st.success(f"Cluster: {result['cluster_id']}, Confidence: {result['confidence']:.3f}")
                st.text(f"Extracted Skills: {result['processed_skills']}")
