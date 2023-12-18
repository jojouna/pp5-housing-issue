import streamlit as st
from src.data_management import load_house_price_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps


def page_sale_price_study_body():
    """
    Display housing price and related attributes
    """

    # load the data
    df = load_house_price_data()

    # hard copy from data analysis notebook
    vars_to_study = ['1stFlrSF', 'GarageArea', 'GarageYrBlt', 'GrLivArea', 
    'KitchenQual', 'MasVnrArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 
    'YearRemodAdd']

    st.write("### Housing Price Correlation Study (BR1)")
    st.info(
        f"* Business Requirement 1: The client is interested in which "
        f"attributes are highly correlated with the housing price. "
        f"The client expects to visualise the correlated variables "
        f"against the sale price."
    )

    # inspect the total housing data (df)
    if st.checkbox("Inspect Housing Data"):
        st.write(
            f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
            f"The following table shows the top 10 rows from the dataset."            
        )

        st.write(df.head(10))

    st.write("---")

    st.write(
        f"* A correlation study was conducted in the Jupyter notebook to "
        f"better understand how the variables are correlated to the house "
        f"price. "
        f"The most correlated attributes are \n\n" 
        f"  *{vars_to_study}*"
    )

    if st.checkbox("Heatmap: Spearman, Pearson and PPS Correlations:"):
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)
        DisplayCorrAndPPS(df_corr_spearman = df_corr_spearman, 
                  df_corr_pearson = df_corr_pearson,               
                  pps_matrix = pps_matrix,
                  CorrThreshold = 0.4, PPS_Threshold = 0.2, 
                  figsize=(12,10), font_annot=10)





# Below functions are to load plots on our dashboard
# that are related with the sale price.
# Codes were hard copied from 03 - Data-Analysis Jupyter notebook.
# plt.show() was replaced to fit the steamlit function to show plots.

def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        fig, axes = plt.subplots(figsize=figsize)
        sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='viridis', 
                    annot_kws={"size": font_annot}, ax=axes,
                    linewidth=0.5
                    )
        axes.set_yticklabels(df.columns, rotation=0)
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[abs(df) < threshold] = True
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                         mask=mask, cmap='rocket_r', 
                         annot_kws={"size": font_annot},
                         linewidth=0.05, linecolor='grey')
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def CalculateCorrAndPPS(df):
    df_corr_spearman = df.corr(method="spearman")
    df_corr_pearson = df.corr(method="pearson")

    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore'])\
        .pivot(columns='x', index='y', values='ppscore')

    pps_score_stats = pps_matrix_raw.query("ppscore < 1").filter(['ppscore'])\
        .describe().T
    print("PPS threshold - check PPS score IQR to decide ")
    print("threshold for heatmap \n")
    print(pps_score_stats.round(3))

    return df_corr_pearson, df_corr_spearman, pps_matrix


def DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix, 
                      CorrThreshold, PPS_Threshold,
                      figsize=(20, 12), font_annot=8):

    print("\n")
    print("* Analyse how the target variable for your ML models are ")
    print("correlated with other variables (features and target)")
    print("* Analyse multi-colinearity, that is, how the features ")
    print("are correlated among themselves")

    print("\n")
    print("*** Heatmap: Spearman Correlation ***")
    print("It evaluates monotonic relationship \n")
    heatmap_corr(df=df_corr_spearman, threshold=CorrThreshold, 
                 figsize=figsize, font_annot=font_annot)

    print("\n")
    print("*** Heatmap: Pearson Correlation ***")
    print("It evaluates the linear relationship between ")
    print("two continuous variables \n")
    heatmap_corr(df=df_corr_pearson, threshold=CorrThreshold, 
                 figsize=figsize, font_annot=font_annot)

    print("\n")
    print("*** Heatmap: Power Predictive Score (PPS) ***")
    print(f"PPS detects linear or non-linear relationships between ")
    print("two columns.\n")
    print(f"The score ranges from 0 (no predictive power) ")
    print("to 1 (perfect predictive power) \n")
          
    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold, 
                figsize=figsize, font_annot=font_annot)