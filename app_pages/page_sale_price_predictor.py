import streamlit as st
import pandas as pd
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

    