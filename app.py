import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import project_summary
from app_pages.page_price_cor_study import page_sale_price_study_body

app = MultiPage(app_name="Housing Price Predictor")

# add app pages
app.add_page("Quick Project Summary", project_summary)
app.add_page("House Price Correlation Study", page_sale_price_study_body)

# run the app
app.run()