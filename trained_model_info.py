import re

from pycaret.classification import load_model as load_model_classification, setup as setup_classification, pull as pull_classification
from pycaret.regression import load_model as load_model_regression, setup as setup_regression, pull as pull_regression
from pycaret.clustering import load_model as load_model_clustering, setup as setup_clustering, pull as pull_clustering

import pandas as pd


def remove_memory_location(text):
    cleaned_text = re.sub(r'Pipeline\(memory=FastMemory\(location=[^\)]+\)', 'Pipeline(memory=FastMemory(location=))', text)
    return cleaned_text


def write_trained_model_info(target, model_type):
    df = pd.read_csv('data/sourcedata.csv')

    if model_type == "Classification":
        setup_classification(data=df, target=target, html=False)
        best_model = load_model_classification('best_model')
        results = pull_classification()

    elif model_type == "Regression":
        setup_regression(data=df, target=target, html=False)
        best_model = load_model_regression('best_model')
        results = pull_regression()

    elif model_type == "Clustering":
        setup_clustering(data=df, html=False)
        best_model = load_model_clustering('best_model')
        results = pull_clustering()

    else:
        raise ValueError("Invalid model_type. Choose from 'Classification', 'Regression', or 'Clustering'.")

    best_model_str = str(best_model)
    best_model_cleaned = remove_memory_location(best_model_str)

    with open('model_info.txt', 'w') as f:
        f.write("Model Information:\n")
        f.write(best_model_cleaned)
        f.write("\n\n")

        f.write("Experiment Results:\n")
        f.write(results.to_string())
        f.write("\n")
