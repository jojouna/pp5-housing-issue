import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import project_summary

app = MultiPage(app_name="Housing Price Predictor")

# add app pages
app.add_page("Quick Project Summary", project_summary)

app.run()