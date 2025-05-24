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

st.set_page_config(page_title="Job Classifier & Notifier", layout="wide")

if "page_index" not in st.session_state:
    st.session_state.page_index = 0

pages = [
    "🔍 Scrape Jobs",
    "🧠 Train Clustering Model",
    "🧪 Classify a Job",
    "📂 Batch Classify",
    "📧 Send Email Alerts"
]

st.sidebar.title("Navigation")
st.sidebar.write("Use arrows to navigate pages")
st.sidebar.write(f"Page {st.session_state.page_index + 1} of {len(pages)}")
if st.sidebar.button("⬅️ Previous"):
    st.session_state.page_index = (st.session_state.page_index - 1) % len(pages)
if st.sidebar.button("➡️ Next"):
    st.session_state.page_index = (st.session_state.page_index + 1) % len(pages)

selection = pages[st.session_state.page_index]
st.title("📊 Job Posting Classifier & Notifier")
st.subheader(selection)

if selection == "🔍 Scrape Jobs":
    keyword = st.text_input("Enter job keyword:", value="data science")
    pages_to_scrape = st.slider("Pages to scrape", 1, 10, 2)
    if st.button("Start Scraping"):
        df = scrape_karkidi_jobs(keyword, pages_to_scrape)
        if not df.empty:
            filename = f"jobs_{keyword.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            st.success(f"✅ Scraped {len(df)} jobs.")
            st.download_button("📁 Download Jobs CSV", df.to_csv(index=False), file_name=filename)
            st.dataframe(df.head())
        else:
            st.warning("⚠️ No jobs found.")

elif selection == "🧠 Train Clustering Model":
    if st.button("Load Latest Jobs & Train Model"):
        df = load_latest_jobs_csv()
        if df is not None and len(df) >= 4:
            df = preprocess_job_data(df)
            df_clustered, model_data, model_filename = train_clustering_model(df)
            clustered_filename = f"jobs_clustered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_clustered.to_csv(clustered_filename, index=False)
            st.success(f"✅ Model: {model_filename}")
            st.success(f"✅ Clustered Jobs: {clustered_filename}")
            st.dataframe(df_clustered.head())
        elif df is not None:
            st.error("❗ Minimum 4 jobs required to cluster.")
        else:
            st.warning("⚠️ Scrape jobs before training.")

elif selection == "🧪 Classify a Job":
    model_data = load_latest_model()
    if model_data:
        with st.form("job_form"):
            title = st.text_input("Job Title")
            company = st.text_input("Company")
            location = st.text_input("Location")
            skills = st.text_input("Skills (comma-separated)")
            summary = st.text_area("Job Description")
            submit = st.form_submit_button("Classify")

        if submit:
            job_data = {'title': title, 'company': company, 'location': location, 'skills': skills, 'summary': summary}
            result = classify_single_job(job_data, model_data)
            if result:
                st.success(f"🔖 Cluster: {result['cluster_id']}")
                st.metric("📈 Confidence", f"{result['confidence']:.3f}")
                st.text_area("Extracted Skills", result['processed_skills'])
            else:
                st.error("❌ Classification failed.")
    else:
        st.warning("⚠️ Train a model first.")

elif selection == "📂 Batch Classify":
    model_data = load_latest_model()
    if model_data:
        uploaded_file = st.file_uploader("Upload jobs CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.info(f"🔄 Classifying {len(df)} jobs...")
            results = classify_jobs_from_csv(uploaded_file.name, model_data)
            if not results.empty:
                output_filename = f"classified_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results.to_csv(output_filename, index=False)
                st.download_button("⬇️ Download Classified Jobs", results.to_csv(index=False), file_name=output_filename)
                st.dataframe(results.head())
                analyze_classification_results(results)
    else:
        st.warning("⚠️ Model not found. Train first.")

elif selection == "📧 Send Email Alerts":
    with st.form("email_form"):
        sender_email = st.text_input("Gmail address")
        sender_password = st.text_input("App password", type="password")
        submitted = st.form_submit_button("Send Alerts")

    if submitted:
        users_df = load_user_preferences()
        clustered_files = glob.glob("jobs_clustered_*.csv")
        if not clustered_files:
            st.error("❌ No clustered job data found.")
        else:
            latest_file = max(clustered_files, key=os.path.getctime)
            jobs_df = load_new_jobs(latest_file)
            if not users_df.empty and not jobs_df.empty:
                match_jobs_to_users(users_df, jobs_df, sender_email, sender_password)
                st.success("📩 Job alerts sent!")
            else:
                st.warning("⚠️ Missing users or job data.")
