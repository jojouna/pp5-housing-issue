import streamlit as st


def predict_sale_price(X_live, regression_pipe, house_features):

    # from live data, subset features related to this pipeline
    X_live_sale_price = X_live.filter(house_features)

    # apply the pipeline to the live data and predict the sale price
    sale_price_prediction = regression_pipe.predict(house_features)

    # create a statement to display the result
    statement = (
        f"* The predicted price for the house based on the "
        f"provided attributes is **{sale_price_prediction.round(0)}**")

    st.write(statement)
