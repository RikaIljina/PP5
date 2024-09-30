import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"The goal was to ..." 
        f"Average Image, Variability Image and Difference between Averages ..."
        f" "

    )

    st.write("### Conclusions and recommendations")
    st.info(
        f"The resulting dark images are an indicator for either a balanced amount of variance / variety or "
        f"high bias in the dataset due to the presence of too many similar images. \n\n"
        f"* Lighting\n"
        f"* Poses\n"
        f"* More pet, less background\n"
        )

    st.write("---")