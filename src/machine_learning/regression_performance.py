import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    """
    Function to show regression performance result.
    Code adapted from Modelling-Evaluation notebook.
    """
    st.write("### Model Evaluation \n")
    st.write("#### Train Set \n")
    regression_evaluation(X_train, y_train, pipeline)
    st.write("#### Test Set \n")
    regression_evaluation(X_test, y_test, pipeline)


def regression_evaluation(X, y, pipeline):
    """
    Function to show regression evaluation result.
    Code adapted from Modelling-Evaluation notebook.
    """
    prediction = pipeline.predict(X)
    st.write(
        'R2 Score:', r2_score(y, prediction).round(3))
    st.write(
        'Mean Absolute Error:', mean_absolute_error(y, prediction).round(3))
    st.write(
        'Mean Squared Error:', mean_squared_error(y, prediction).round(3))
    st.write(
        'Root Mean Squared Error:', np.sqrt(mean_squared_error(y,
                                            prediction)).round(3))
    st.write("\n")


def regression_evaluation_plots(
     X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5):
    """
    Function to display the regression evaluation plots.
    Code adapted from Modelling-Evaluation notebook.
    """
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    sns.scatterplot(x=y_train['SalePrice'],
                    y=pred_train, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train['SalePrice'],
                 y=y_train['SalePrice'], color='red', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("* Train Set")

    sns.scatterplot(x=y_test['SalePrice'], y=pred_test,
                    alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test['SalePrice'], y=y_test['SalePrice'],
                 color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("* Test Set")

    st.pyplot(fig)
