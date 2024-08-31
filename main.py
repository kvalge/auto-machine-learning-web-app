import streamlit as st
import pandas as pd
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup as setup_classification, compare_models as compare_models_classification, pull as pull_classification, save_model as save_model_classification
from pycaret.regression import setup as setup_regression, compare_models as compare_models_regression, pull as pull_regression, save_model as save_model_regression
from pycaret.clustering import setup as setup_clustering, create_model as create_model_clustering, pull as pull_clustering, save_model as save_model_clustering

from trained_model_info import write_trained_model_info

target = ''


with st.sidebar:
    st.image("static/image.png")
    st.title("Auto Machine Learning App")
    st.info("To upload the dataset, profile data, train the model and download the best model.")

    choice = st.radio("Functionalities", ["Upload", "Profiling", "Training", "Download"])

if os.path.exists("data/sourcedata.csv"):
    df = pd.read_csv("data/sourcedata.csv", index_col=None)


if choice == "Upload":
    st.title("Upload Data For Modelling")
    file = st.file_uploader("Upload the dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df = df.dropna()
        df.to_csv("data/sourcedata.csv", index=None)
        st.dataframe(df)


if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)


if choice == "Training":
    st.title("Machine Learning")
    model_type = st.radio("Select model type", ["Classification", "Regression", "Clustering"])

    if model_type == "Classification":
        target = st.selectbox("Select target", df.columns)

        if st.button("Start Model Comparison"):
            setup_classification(df, target=target)
            setup_df = pull_classification()
            st.info("ML settings")
            st.dataframe(setup_df)

            best_model = compare_models_classification()
            compare_df = pull_classification()
            st.info("ML model")
            st.dataframe(compare_df)

            st.success("Best Model: {}".format(best_model))
            save_model_classification(best_model, 'best_model')

            write_trained_model_info(target, model_type)

    elif model_type == "Regression":
        target = st.selectbox("Select target", df.columns)

        if st.button("Start Model Comparison"):
            setup_regression(df, target=target)
            setup_df = pull_regression()
            st.info("ML settings")
            st.dataframe(setup_df)

            best_model = compare_models_regression()
            compare_df = pull_regression()
            st.info("ML model")
            st.dataframe(compare_df)

            st.success("Best Model: {}".format(best_model))
            save_model_regression(best_model, 'best_model')

            write_trained_model_info(target, model_type)

    elif model_type == "Clustering":
        if st.button("Start Clustering"):
            setup_clustering(df)
            setup_df = pull_clustering()
            st.info("Clustering settings")
            st.dataframe(setup_df)

            model = create_model_clustering('kmeans')
            st.info(f"Clustering Model Created: {model}")
            compare_df = pull_clustering()
            st.info("Clustering model")
            st.dataframe(compare_df)

            st.success("Clustering Model Created: {}".format(model))
            save_model_clustering(model, 'best_model')

            write_trained_model_info(target, model_type)


if choice == "Download":
    st.title("Download The Best Model")
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download the Model", f, "trained_model.pkl")
    else:
        st.warning("No model found. Please run the ML section first.")







