import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import glob
import os
from job_utils import (
    scrape_karkidi_jobs,
    preprocess_job_data,
    train_clustering_model,
    load_latest_model,
    classify_single_job,
    classify_jobs_from_csv,
    analyze_classification_results,
    load_latest_jobs_csv,
    load_user_preferences,
    load_new_jobs,
    match_jobs_to_users
)

st.set_page_config(page_title="Job Classifier", layout="wide")
st.title("🔍 Job Posting Classifier and Clustering App")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Scrape Jobs", 
    "Train Clustering", 
    "Classify Job", 
    "Batch Classify", 
    "Send Email Alerts"
])

# Tab 1: Scrape Jobs
with tab1:
    st.header("Step 1: Scrape Job Listings")
    keyword = st.text_input("Enter keyword", value="data science")
    pages = st.slider("Pages to scrape", 1, 10, 2)
    if st.button("Scrape Jobs"):
        df = scrape_karkidi_jobs(keyword, pages)
        if not df.empty:
            filename = f"jobs_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            st.success(f"Scraped {len(df)} jobs.")
            st.download_button("Download CSV", df.to_csv(index=False), file_name=filename)
            st.dataframe(df.head())
        else:
            st.warning("No jobs found.")

# Tab 2: Train Clustering
with tab2:
    st.header("Step 2: Train K-means Clustering")
    if st.button("Train on Latest Jobs CSV"):
        df = load_latest_jobs_csv()
        if df is not None and len(df) >= 4:
            df = preprocess_job_data(df)
            df_clustered, model_data, model_filename = train_clustering_model(df)
            clustered_filename = f"jobs_clustered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_clustered.to_csv(clustered_filename, index=False)
            st.success(f"Model saved: {model_filename}")
            st.success(f"Clustered jobs saved: {clustered_filename}")
            st.dataframe(df_clustered.head())
        else:
            st.warning("Not enough jobs to cluster.")

# Tab 3: Classify a Single Job
with tab3:
    st.header("Step 3: Classify a Job")
    model_data = load_latest_model()
    if model_data:
        title = st.text_input("Job Title")
        company = st.text_input("Company")
        location = st.text_input("Location")
        skills = st.text_input("Skills (comma-separated)")
        summary = st.text_area("Summary")

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
                st.success(f"Cluster ID: {result['cluster_id']}")
                st.info(f"Confidence: {result['confidence']:.3f}")
                st.text(f"Extracted Skills: {result['processed_skills']}")
            else:
                st.error("Classification failed.")
    else:
        st.warning("No trained model found. Please train first.")

# Tab 4: Batch Classify from CSV
with tab4:
    st.header("Step 4: Batch Classify Jobs")
    model_data = load_latest_model()
    if model_data:
        uploaded_file = st.file_uploader("Upload jobs CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            results = classify_jobs_from_csv(uploaded_file.name, model_data)
            if not results.empty:
                output_filename = f"classified_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results.to_csv(output_filename, index=False)
                st.download_button("Download Results", results.to_csv(index=False), file_name=output_filename)
                st.dataframe(results.head())
                analyze_classification_results(results)
    else:
        st.warning("No trained model available.")

# Tab 5: Send Email Alerts
with tab5:
    st.header("Step 5: Send Job Alerts to Users")
    sender_email = st.text_input("Sender Gmail")
    sender_password = st.text_input("App Password", type="password")

    if st.button("Send Alerts"):
        users_df = load_user_preferences()
        clustered_files = glob.glob("jobs_clustered_*.csv")
        if clustered_files:
            latest_file = max(clustered_files, key=os.path.getctime)
            jobs_df = load_new_jobs(latest_file)
            if not users_df.empty and not jobs_df.empty:
                match_jobs_to_users(users_df, jobs_df, sender_email, sender_password)
                st.success("Job alerts sent successfully.")
            else:
                st.warning("Missing user or job data.")
        else:
            st.error("No clustered job files found.")
