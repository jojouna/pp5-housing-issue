import streamlit as st


def page_project_hypothesis():
    """
    Display project hypothesis after correlation study
    """

    st.write("### Project Hypothesis")

    st.success(
        f"The conclusion we could make from our observations are: \n\n"
        f"* Size matters: 1stFlrSF, GarageArea, GrLivArea, TotalBsmtSF shows "
        f"that generally when the size is bigger, the sale price grows higher."
        f"\n"
        f"  * We can make our hypothesis 1 as **The size of a house positively "
        f"correlates with the sale price**\n"
        f"* Remodeling year matters: YearRemodAdd shows when there was a "
        f"recent remodel with the house, the sale price is higher."
        f"\n"
        f"  * We can make our hypothesis 2 as **Year of the house remodeling "
        f"positively correlates with the sale price**\n"
        f"* Quality matters: KitchenQual and OverallQual shows that houses "
        f"with higher quality have a higher sale price.\n"
        f"  * We can make our hypothesis 3 as **The quality of the house "
        f"positively correlates with the sale price**"
    )