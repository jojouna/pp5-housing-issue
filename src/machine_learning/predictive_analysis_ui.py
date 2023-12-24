import streamlit as st


def predict_sale_price(X_live, regression_pipe, house_features):
    """
    Function to predict a sale price with the selected house
    features and the regression pipeline.
    sale_price_prediction is transformed to a string
    so that is shows clear with not brackets around the
    numbers. Statement will be shown.
    """

    X_live_sale_price = X_live.filter(house_features)

    sale_price_prediction = regression_pipe.predict(X_live_sale_price)

    sale_price_prediction_str = str(int(sale_price_prediction.round(0)))

    statement = (
        f"* The predicted price for the house based on the "
        f"provided attributes is $**{sale_price_prediction_str}**")

    st.success(statement)
