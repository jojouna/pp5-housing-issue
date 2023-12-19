import streamlit as st
import pandas as pd
import numpy as np
from src.data_management import (
    load_house_price_data,
    load_inherited_house_data,
    load_pkl_file
)


def page_sale_price_predictor():

    version = "v1"

    # load predict price files
    regression_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_saleprice/{version}/clf_pipeline.pkl")
    house_features = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv")
    df_inherited_houses = load_inherited_house_data()

    st.write("### Predict Sale Price of Inherited Houses (BR2)")

    st.info(
        f"* Business Requirement 2: The client is interested in predicting the "
        f"sale price for her 4 inherited houses, and any other sale price "
        f"in Ames, Iowa."
    )

    st.write(
        f"* The below table shows the features of 4 inherited houses."
    )

    st.write(df_inherited_houses)

    st.write(
        f"* We set only with the most relevant features for the 4 "
        f"inherited houses, and display the predicted sale price for each "
        f"house."
    )

    df_inherited_relevant_features = df_inherited_houses.filter(house_features)

    st.write(df_inherited_relevant_features)
    sale_price_predict = regression_pipe.predict(df_inherited_relevant_features)
    # numpy predict function reference: 
    # https://www.askpython.com/python/examples/python-predict-function
    
    st.write(
        f"* Predicted sale price for each 4 inherited house:"
    )
    st.success(
        f"We apply the attributes of 4 inherited houses to our pipeline model "
        f"that has been already created through investigation."
    )

    st.write(sale_price_predict.round(0))

