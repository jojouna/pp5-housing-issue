import streamlit as st 

def project_summary():
    st.write("### Quick Project Summary")

    st.info(
        f"Housing Price Predictor is a project developed to predict housing "
        f"price in Ames, Iowa, USA based on various attributes of the house."
        f"\n\n"
        f"**Project terms and Jargon**\n"
        f"* **Sale price** is the current market price of the house with  "
        f"various attributes.\n"
        f"The currency is all set to US Dollar.\n"
        f"* **Inherited house** is the house that the client inherited from "
        f"their grandparents.\n\n"
        f"* The units that are used for house measurements are all in "
        f"**square feet**.\n\n"
        f"* For additional jargon related with housing attributes please "
        f"check [Dataset Content](https://github.com/choyoon88/pp5-housing"
        "-issue?tab=readme-ov-file#dataset-content) on my Readme.\n\n"
        f"**Project Dataset**\n"
        f"* The project dataset comes from housing price database in "
        f"[Kaggle](https://www.kaggle.com/datasets/"
        f"codeinstitute/housing-prices-data) "
        f"created by Code Institute.\n"
        f"* The data represents the housing price in Ames, Iowa, USA with 23"
        f" aspects of the house such as the size of the house, built year etc."
        f" The total number of houses is 1460."
    )

    # Link to project README file for more information of the project
    st.write(
        f"For additional information, please visit the "
        f"[Project Readme File](https://github.com/choyoon88/"
        f"pp5-housing-issue/)."
    )

    st.success(
        f"**Business Requirements**\n\n"
        f"The project has two major business requirements.\n\n"
        f"* BR1: The client is interested in discovering the most relevant "
        f"variable that correlates with the sale price.\n"
        f"* BR2: The client wants to have a predicting model of the 4 inherited "
        f"houses, as well as any other houses in Ames, Iowa.\n\n"
    )