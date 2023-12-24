import streamlit as st
import pandas as pd
import numpy as np
from src.data_management import (
    load_house_price_data,
    load_inherited_house_data,
    load_pkl_file
)
from src.machine_learning.predictive_analysis_ui import predict_sale_price


def page_sale_price_predictor():
    """
    Part 1: Create a sale price prediction of the client's
    four inherited houses by implementing the data of the
    inherited houses' attribute values to the regression pipe.
    Part 2: Create a predict sale price widget that has the
    input areas of the relevant attributes and display
    the predicted price by passing the input values to the
    regression pipe.
    """
    version = "v1"

    regression_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_saleprice/{version}/clf_pipeline.pkl")
    house_features = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv")
    df_inherited_houses = load_inherited_house_data()

    st.write("### Predict Sale Price of Inherited Houses (BR2 Part 1)")

    st.info(
        f"* Business Requirement 2: The client is interested in predicting "
        f"the sale price for her 4 inherited houses, and any other sale "
        f"price in Ames, Iowa."
    )

    st.write(
        f"* The below table shows the features of 4 inherited houses."
    )

    st.write(df_inherited_houses)

    st.write(
        f"* We set only with the most relevant features as an attribute "
        f"for the 4 inherited houses, and display the predicted sale price "
        f"for each house."
    )

    df_inherited_relevant_features = df_inherited_houses.filter(house_features)

    st.write(df_inherited_relevant_features)

    sale_price_predict = regression_pipe.predict(
        df_inherited_relevant_features)
    """
    Numpy predict function reference:
    https://www.askpython.com/python/examples/python-predict-function
    """

    sale_price_predict_df = pd.DataFrame(sale_price_predict,
                                         columns=["Predicted Sale Price"])

    st.write(
        f"* Predicted sale price for each 4 inherited house:"
    )

    st.success(
        f"We apply the attributes of 4 inherited houses to our pipeline model "
        f"that has been already created through investigation."
    )

    st.write(sale_price_predict_df.round(0))

    price_sum = sale_price_predict_df["Predicted Sale Price"].sum()
    st.success(
        f"The sum of predicted sale price for 4 houses is ${price_sum.round()}"
        )

    st.write("---")

    st.write("### Predict Sale Price of Ames, Iowa (BR2 Part 2)")

    st.info(
        f"* Business Requirement 2: The client is interested in predicting "
        f"sale price for any house in Ames, Iowa with the attributes that are "
        f"highly correlated with the sale price.\n"
    )

    st.write(
        f"* We have 4 attributes that are strongly correlated with sale "
        f"price. Below is the function that calculates and predicts the "
        f"sale price according to the values of the 4 attributes.\n"
        f"* The client could now enter the values of the attribute to predict "
        f"the sale price of the house."
    )

    X_live = DrawInputsWidget()

    df = load_house_price_data()

    if st.button("Predict Sale Price"):
        price_prediction = predict_sale_price(
            X_live, regression_pipe, house_features
        )


def DrawInputsWidget():
    """
    Define inputs widget so that the user can enter attributes of the
    house so that they can predict the sale price.
    """

    df = load_house_price_data()
    percentageMin, percentageMax = 0.4, 2.0

    col1, col2 = st.beta_columns(2)
    col3, col4 = st.beta_columns(2)

    X_live = pd.DataFrame([], index=[0])

    with col1:
        feature = '2ndFlrSF'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax
        )
    X_live[feature] = st_widget

    with col2:
        feature = 'GarageArea'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax
        )
    X_live[feature] = st_widget

    with col3:
        feature = 'OverallQual'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax
        )
    X_live[feature] = st_widget

    with col4:
        feature = 'TotalBsmtSF'
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax
        )
    X_live[feature] = st_widget

    return X_live
