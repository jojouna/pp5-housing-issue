import streamlit as st
from src.data_management import load_house_price_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps


# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)


def page_sale_price_study_body():
    """
    Display housing price and related attributes
    """

    df = load_house_price_data()

    vars_to_study = ['1stFlrSF', 'GarageArea', 'GarageYrBlt', 'GrLivArea',
                     'KitchenQual', 'MasVnrArea', 'OverallQual', 'TotalBsmtSF',
                     'YearBuilt', 'YearRemodAdd']

    st.write("### Housing Price Correlation Study (BR1)")
    st.info(
        f"* Business Requirement 1: The client is interested in which "
        f"attributes are highly correlated with the sale price. "
        f"The client expects to visualise the correlated variables "
        f"against the sale price."
    )

    if st.checkbox("Inspect Housing Data"):
        st.write(
            f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
            f"The following table shows the top 10 rows from the dataset.")

        st.write(df.head(10))

    st.write("---")

    st.write(
        f"* A correlation study was conducted in the Jupyter notebook to "
        f"better understand how the variables are correlated to the sale "
        f"price. "
        f"The most correlated attributes are \n\n"
        f"  *{vars_to_study}*"
    )

    st.info(
        f"* The below heatmaps were created to highlight the correlation "
        f"between variables. Spearman, Pearson and Predictive Power Score "
        f"(PPS) were used to create 3 different heatmaps to have a various of "
        f"perspectives to check which variables are more correlated to the "
        f"sale price.\n\n"
        f"* Each plots are Spearman, Pearson and PPS correlation plots "
        f"respectively."
    )

    if st.checkbox("Heatmap: Spearman, Pearson and PPS Correlations:"):
        df_corr_spearman, df_corr_pearson, pps_matrix = CalculateCorrAndPPS(df)
        DisplayCorrAndPPS(df_corr_spearman=df_corr_spearman,
                          df_corr_pearson=df_corr_pearson,
                          pps_matrix=pps_matrix,
                          CorrThreshold=0.4, PPS_Threshold=0.2,
                          figsize=(12, 10), font_annot=10)

    st.info(
        f"After we have conducted correlation studies, we figured out some of "
        f"the variables that were highly correlated with the sale price. "
        f"We can now display the plots of each variables that were "
        f"strongly correlated with the sale price according to the type of "
        f"the variables.\n\n"
        f"The summary of the plots is, \n\n"
        f"* Target variable is positively skewed: 1stFlrSF, GrLivArea, "
        f"MasVnrArea and TotalBsmtSF have most of sale price clustered on the "
        f"left side of the plot.\n"
        f"* The size of the house is correlated with the sale price. "
        f"1stFlrSF, GarageArea, GrLivArea, TotalBsmtSF shows that generally "
        f"when the size is bigger, the sale price grows higher.\n"
        f"* Recently being remodelled makes the sale price grow up. "
        f"YearRemodAdd shows when there was a recent remodel with the house, "
        f"the sale price is higher.\n "
        f"* Better quality of the house leads to a higher sale price. "
        f"KitchenQual and OverallQual shows that houses with higher quality "
        f"have a higher sale price."
    )

    df_eda = df.filter(vars_to_study + ['SalePrice'])
    target_var = 'SalePrice'

    st.write("### Data Visualisation")

    if st.checkbox("Sale price by variables"):
        sale_price_by_variables_plot(df_eda)


def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to create heatmap using correlation.
    Codes were adapted from data-analysis jupyter notebook.
    From the original codes, plt.show() was replaced to fit the 
    streamlit function to show plots.
    """
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
    """
    Function to create heatmap using pps
    """
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
    """
    Function to calculate correlation and pps
    """
    df_corr_spearman = df.corr(method="spearman")
    df_corr_pearson = df.corr(method="pearson")

    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore'])\
        .pivot(columns='x', index='y', values='ppscore')

    pps_score_stats = pps_matrix_raw.query("ppscore < 1").filter(['ppscore'])\
        .describe().T
    print(pps_score_stats.round(3))

    return df_corr_spearman, df_corr_pearson, pps_matrix


def DisplayCorrAndPPS(df_corr_spearman, df_corr_pearson, pps_matrix,
                      CorrThreshold, PPS_Threshold,
                      figsize=(20, 12), font_annot=8):
    """
    Function to display correlation and pps
    """
    heatmap_corr(df=df_corr_spearman, threshold=CorrThreshold,
                 figsize=figsize, font_annot=font_annot)

    heatmap_corr(df=df_corr_pearson, threshold=CorrThreshold,
                 figsize=figsize, font_annot=font_annot)

    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold,
                figsize=figsize, font_annot=font_annot)


def sale_price_by_variables_plot(df):
    """
    Create a three different kind of plots of the 
    top 10 variables that correlate with the sale price.
    Plots differ by the kind of variables.
    """
    numerical_vars = ['1stFlrSF', 'GarageArea', 'GrLivArea',
                      'MasVnrArea', 'TotalBsmtSF']
    categorical_vars = ['KitchenQual', 'OverallQual']
    time_vars = ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd']

    target_var = 'SalePrice'

    for col in df.columns:
        if col == target_var:
            continue

        if col in categorical_vars:
            fig, axes = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x=col, y=target_var)
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.ylabel(target_var)
            st.pyplot(fig)

        elif col in numerical_vars:
            fig, axes = plt.subplots(figsize=(8, 6))
            sns.lmplot(data=df, x=col, y=target_var, ci=None)
            plt.title(f"LMplot for {col}")
            st.pyplot()

        elif col in time_vars:
            fig, axes = plt.subplots(figsize=(8, 6))
            sns.lineplot(data=df, x=col, y=target_var)
            plt.title(f"Lineplot for {col}")
            plt.xlabel(col)
            plt.ylabel(target_var)
            st.pyplot(fig)
