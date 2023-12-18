import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load housing data with all attributes
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_house_price_data():
    df = pd.read_csv("outputs/datasets/collection/house_prices_ames_iowa.csv")
    return df


# load inherited house data
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_inherited_house_data():
    df_inherited = pd.read_csv(
        "outputs/datasets/collection/inherited_houses.csv"
        )
    return df_inherited


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)