import streamlit as st
import pandas as pd
from collections import Counter

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import run_classification
from src.utils import process_inputs, update_info_box


def page_image_classifier_body():

    st.write("### Image Classifier")

    st.markdown(
        """
        <div class="blue-div">
          <h5>Business requirement 2 and 3:</h5>
          <p>The client is interested in a proof-of-concept model that will
          tell pets apart by their images and achieves an 
          <span class="green-bg">F1 score > 0.9 for each label.</span><br>
          The client would like to investigate the possibility of an infallible
          process during which a pet will be either classified correctly or not
          classified at all.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("\n")

    st.markdown(
        """
        <div class="blue-div">
            <h5>Answering business requirement 2 and 3:</h5>
            <p>The following classification tool uses the developed model to
                analyze and classify any given image with the correct
                parameters.<br>
                The tool will aggregate images into batches and try to predict
                the main class present until it reaches the preset confidence
                threshold or exhausts its preset maximum attempt limit.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("\n\n")

    st.markdown(
        """
        <div class="yellow-div">
            <p>
            You can download a set of pet images for live prediction from 
            Google Drive: 
            <a href="https://drive.usercontent.google.com/u/0/uc?id=1M4vruKofgkxSTwYd1FCdtoajfvmZFP6Y&export=download" 
            target="_blank" rel="noopener">Live images</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("\n\n")

    st.write("---")
    st.markdown(
        """
        <div class="blue-div">
        <b>Please provide parameters for the classification.</b><br>
        <ul>
        <li>
        A <b>trial</b> is a series of classification attempts from individual
        images that are assumed to belong to the same class. The final
        classification result is based on the mean values of all classification
        attempts within that trial.
        </li>
        <li>An <b>attempt</b> is the model's mean probability calculation from
        all probabilities gathered so far within a single trial and the
        comparison of the top value to the confidence threshold.
        </li>
        </ul>
        <p>The following parameters may be clipped or adjusted based on the
        amount of provided images.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("---")

    if st.session_state.get("tr_input") or st.session_state.get("update_str"):
        update_str = st.session_state["update_str"]
    else:
        update_str = ""
        st.session_state["update_str"] = update_str

    with st.form(key="param_input_form"):
        col1, col2, buff = st.columns([6, 3, 1])
        col1.markdown(
            f'<p style="padding-top: 1rem;">How many consecutive trials should'
            f' the model run?</p>',
            unsafe_allow_html=True,
        )
        trial_amount = col2.number_input(
            label="Amount of trials to run", min_value=1, key="tr_input"
        )
        col1, col2, buff = st.columns([6, 3, 1])
        col1.markdown(
            f'<p style="padding-top: 1rem;">How many images within one trial '
            f'should the model analyze before '
            f"trying to reach a conclusion?</p>",
            unsafe_allow_html=True,
            help=("WARNING: A too low number can lead to misclassification due"
                  "to a high-confidence error early on"),
        )

        min_attempts = col2.number_input(
            label="Minimal image threshold", min_value=1, value=5
        )
        col1, col2, buff = st.columns([6, 3, 1])
        col1.markdown(
            f'<p style="padding-top: 1rem;">After how many images should the '
            f'model give up the classification '
            f'attempt and restart the trial?</p>',
            unsafe_allow_html=True,
        )
        max_attempts = col2.number_input(
            label="Max attempts per trial", min_value=1, value=10
        )
        col1, col2, buff = st.columns([6, 3, 1])
        col1.markdown(
            f'<p style="padding-top: 1rem;">What confidence level should count'
            f' as a successful classification?',
            unsafe_allow_html=True,
        )
        min_confidence = col2.slider(
            label="Confidence threshold", min_value=0, max_value=100, value=75
        )

        submitted_params = st.form_submit_button("SAVE")

        warning_box = st.empty()

        if submitted_params:
            update_str, max_attempts = process_inputs(
                trial_amount,
                min_attempts,
                max_attempts,
                min_confidence,
                warning_box,
            )
            st.session_state['max_attempt'] = max_attempts

    update_box = st.empty()
    update_box.info(update_str)

    st.write("---")

    start_model = False

    clear = not st.checkbox("Keep files in cache")
    shuffle = st.checkbox("Shuffle uploaded files")
    repeat = False
    if shuffle:
        repeat = st.checkbox("Allow repeating images from buffer")

    with st.form("img-upload-form", clear_on_submit=clear):
        images_buffer = st.file_uploader(
            "Upload pet images. You may select more than one.",
            type="png",
            accept_multiple_files=True,
        )
        submitted_img = st.form_submit_button("START CLASSIFICATION")

        if submitted_img and images_buffer is not None:
            start_model = True

    if start_model:

        update_info_box(
            trial_amount,
            min_attempts,
            st.session_state['max_attempt'],
            min_confidence,
            update_box,
        )

        st.write("Loading the model ...")

        result, fav_class_min = run_classification(
            trial_amount,
            min_attempts,
            st.session_state['max_attempt'],
            min_confidence,
            images_buffer,
            shuffle,
            repeat,
        )

        column_list = ["Attempts", "Predicted class", "Confidence"]
        df = pd.DataFrame.from_dict(
            result, orient="index", columns=column_list
        )
        df.index.name = "Trial nr"

        if not df.empty:
            st.success("### **Analysis Report**")
            st.write(f"#### Confidence was set to **{min_confidence}%**")
            cap1 = (f"Batch classification results after {len(df)} Trials "
                    f"with {len(images_buffer)} individual images to choose "
                    f"from")
            st.write(cap1)

            st.dataframe(
                df.style.format({"Confidence": "{:.2f}".format}).apply(
                    lambda x: (
                        ["background-color: #FCF8AE"] * len(x)
                        if not isinstance(x.loc["Attempts"], int)
                        else [""] * len(x)
                    ),
                    axis=1,
                )
            )

            df_favs_maj = pd.DataFrame.from_dict(
                Counter(df["Predicted class"]),
                columns=["Amount"],
                orient="index",
            )
            df_favs_maj.index.name = "Class"

            df_favs_sg = pd.DataFrame.from_dict(
                Counter(fav_class_min), columns=["Amount"], orient="index"
            )
            df_favs_sg.index.name = "Class"

            col1, col2 = st.columns(2)
            with col1:
                cap2 = "Sum of the final conclusions made by the model"
                st.caption(cap2)
                st.dataframe(df_favs_maj)
            with col2:
                cap3 = "Sum of individual predictions for each image"
                st.caption(cap3)
                st.dataframe(df_favs_sg)

            st.write("\n")
            st.markdown(
                download_dataframe_as_csv(
                    (df, df_favs_maj, df_favs_sg), (cap1, cap2, cap3)
                ),
                unsafe_allow_html=True,
            )
