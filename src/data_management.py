import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_house_price_data():
    """
    Load raw housing data with all attributes
    """
    df = pd.read_csv("outputs/datasets/collection/house_prices_ames_iowa.csv")
    return df


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_inherited_house_data():
    """
    Load four inherited houses' data
    """
    df_inherited = pd.read_csv(
        "outputs/datasets/collection/inherited_houses.csv"
        )
    return df_inherited


def load_pkl_file(file_path):
    """
    Load the Pickle file
    """
    return joblib.load(filename=file_path)
