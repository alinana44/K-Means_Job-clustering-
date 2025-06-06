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

st.set_page_config(page_title="Job Clustering and Classification", layout="wide")
st.title("Job Posting Classifier and Clusterer")

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select a Step", [
    "Scrape Job Listings",
    "Train Clustering Model",
    "Classify a Job",
    "Send Email Alerts"
])

# Scrape Jobs
if menu == "Scrape Job Listings":
    st.header("Step 1: Scrape Job Listings")
    with st.form("scrape_form"):
        keyword = st.text_input("Enter job search keyword", value="data science")
        pages = st.slider("Select number of pages to scrape", min_value=1, max_value=10, value=2)
        submit_scrape = st.form_submit_button("Scrape Jobs")

    if submit_scrape:
        with st.spinner("Scraping job postings..."):
            df = scrape_karkidi_jobs(keyword, pages)
        if not df.empty:
            st.success(f"Successfully scraped {len(df)} job listings.")
            filename = f"jobs_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            st.download_button("Download Job Listings CSV", df.to_csv(index=False), file_name=filename)
            st.dataframe(df.head())
        else:
            st.warning("No jobs found. Please try again with a different keyword or check your internet connection.")

# Train Clustering
elif menu == "Train Clustering Model":
    st.header("Step 2: Train Clustering Model")
    if st.button("Load Latest Jobs and Train Model"):
        with st.spinner("Loading latest job listings and training the clustering model..."):
            df = load_latest_jobs_csv()
            if df is not None and len(df) >= 4:
                df = preprocess_job_data(df)
                df_clustered, model_data, model_filename = train_clustering_model(df)
                clustered_filename = f"jobs_clustered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_clustered.to_csv(clustered_filename, index=False)
                st.success(f"Clustering complete. Model saved as: {model_filename}")
                st.download_button("Download Clustered Jobs CSV", df_clustered.to_csv(index=False), file_name=clustered_filename)
                st.dataframe(df_clustered.head())
            elif df is not None:
                st.warning("Not enough job listings to perform clustering. Minimum of 4 required.")
            else:
                st.warning("No job listings found. Please scrape job listings first.")

# Classify a Job
elif menu == "Classify a Job":
    st.header("Step 3: Classify a Job")
    model_data = load_latest_model()
    if model_data:
        with st.form("classification_form"):
            title = st.text_input("Job Title")
            company = st.text_input("Company")
            location = st.text_input("Location")
            skills = st.text_input("Key Skills (comma-separated)")
            summary = st.text_area("Job Description or Summary")
            classify_btn = st.form_submit_button("Classify Job")

        if classify_btn:
            job_data = {
                'title': title,
                'company': company,
                'location': location,
                'skills': skills,
                'summary': summary
            }
            with st.spinner("Classifying job posting..."):
                result = classify_single_job(job_data, model_data)
            if result:
                st.success("Job successfully classified.")
                st.write(f"**Predicted Cluster:** {result['cluster_id']}")
                st.write(f"**Confidence Score:** {result['confidence']:.3f}")
                st.write("**Extracted Skills:**")
                st.code(result['processed_skills'], language='text')
            else:
                st.error("Failed to classify the job. Please make sure all required fields are filled in correctly.")
    else:
        st.warning("No trained clustering model found. Please train a model before classification.")

# Send Email Alerts
elif menu == "Send Email Alerts":
    st.header("Step 4: Send Job Alerts via Email")
    with st.form("email_form"):
        sender_email = st.text_input("Sender Gmail Address")
        sender_password = st.text_input("Gmail App Password", type="password")
        submit_alerts = st.form_submit_button("Send Alerts")

    if submit_alerts:
        with st.spinner("Sending job alerts to users..."):
            users_df = load_user_preferences()
            clustered_files = glob.glob("jobs_clustered_*.csv")
            if clustered_files:
                latest_file = max(clustered_files, key=os.path.getctime)
                jobs_df = load_new_jobs(latest_file)
                if not users_df.empty and not jobs_df.empty:
                    match_jobs_to_users(users_df, jobs_df, sender_email, sender_password)
                    st.success("Job alerts sent successfully.")
                else:
                    st.warning("Missing user preferences or job listings.")
            else:
                st.error("No clustered job file found. Please cluster jobs first.")
