import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.data_management import load_pkl_file
from src.machine_learning.regression_performance import (
    regression_performance,
    regression_evaluation_plots
)


def page_ml_predict_price():
    """
    Display ML pipeline and feature importance plit
    """

    # load the data
    version = 'v1'
    regression_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_saleprice/{version}/clf_pipeline.pkl")
    feature_importance_plot = plt.imread(
        f"outputs/ml_pipeline/predict_saleprice"
        f"/{version}/feature_importance_plot.png"
    )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv"
    )
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/X_test.csv"
    )
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/y_train.csv"
    )
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/y_test.csv"
    )

    st.write(
        f"### ML: Predict Sale Price"
    )

    # summary of the page
    st.info(
        f"* To meet the business requirement 2, we trained a regression model "
        f"to predict the sale price of the houses in Ames, Iowa. The pipeline "
        f"showed the R2 score of at least 0.75 on both train and test sets.\n"
        f"* The pipeline performance for the best model showed R2 scores for "
        f"each train set and test set for **0.86** and **0.77**\n"
        f"* Following is the pipeline steps we took to reach the above "
        f"conclusion."
    )
    st.write("---")

    st.info(
        f"* After going through three pipeline tests, we have decided to "
        f"move ahead with GradientBoostingRegressor which showed the highest "
        f"performance. With the related details we have created a single "
        f"pipeline combining the data cleaning, feature engineering, feature "
        f"scaling and modelling."
    )

    st.code(regression_pipe)
    st.write("---")

    st.info(
        f"* See below for the features and their importance."
    )
    st.write(
        f"* Best Features:"
    )
    st.write(X_train.columns.to_list())
    st.write(
        f"* Importance for Best Features:"
    )
    st.image(feature_importance_plot)
    st.write("---")

    st.write(
        f"### Pipeline Performance"
    )
    st.info(
        f"* We agreed with the client to have the R2 score of at least 0.75 "
        f"on train and test set.\n"
        f"* Our pipeline shows that our model performance have been met."
    )
    regression_performance(X_train=X_train, y_train=y_train,
                           X_test=X_test, y_test=y_test,
                           pipeline=regression_pipe)

    st.write(
        f"### Regression Performance Plot"
    )
    st.info(
        f"* The regression performance plots show that our model predicts "
        f"the sale price in most of the situation. However, it can be "
        f"ineffective with a higher price range."
    )
    regression_evaluation_plots(X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test,
                                pipeline=regression_pipe)
