import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_dataset_assessment import page_dataset_assessment_body
from app_pages.page_image_classifier import page_image_classifier_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics

app = MultiPage(app_name="PetFeeder")  # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Dataset Assessment", page_dataset_assessment_body)
app.add_page("Image Classifier", page_image_classifier_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()  # Run the app
