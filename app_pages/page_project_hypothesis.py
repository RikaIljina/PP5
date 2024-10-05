import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    st.info(
        """
        The goals of this project were:\n
        - to assess image data prior to training to determine weak points in a
        training set and give actionable recommendations to the client
        - to investigate the possibility of a thorough assessment of image data
        using various image comparison methods and predict which labels will be
        hardest to distinguish
        - to train a model that will achieve an F1 score of > 0.9 for test and
        live data for each label
        - to reduce the risk of misclassifying a pet by letting the model make
        a summary prediction for a batch of images instead of trying to
        classify a pet based on one single image
        - to determine an optimal minimum batch size that will make sure that
        an early high-confidence error will not result in the
        misclassification of the batch
        - to determine a reasonable confidence value that will make batch
        misclassification highly unlikely while avoiding false negatives for
        classes with the lowest recall
        - to determine an appropriate upper limit for the batch size after
        which an accurate classification should be guaranteed or the trial
        abandoned due to inconclusive input
        """
    )

    st.write("### Have the Business Requirements been met?")
    st.markdown(
        '''
        <div class="green-div">
          <ol>
            <li>
            The client is interested in a recommendation regarding the scale
            and quality of future datasets as well as an investigation of a
            correlation between the similarity of the pets' visual features and
            the performance of the model.
            <br>
            <br>
            ✅ The requirement has been met by conducting a thorough assessment
            of the image data and summarizing the results on the page "Dataset
            Assessment" and "Recommendations".
            </li>
            <hr>
            <li>
            The client is interested in a PoC model that will tell pets apart
            by their images and achieves an F1 score > 0.9 for each label.
            <br>
            <br>
            ✅ The requirement has been met by developing a fully functional,
            reasonably sized PoC model that meets all target scores.<br>
            The model can be tested by uploading live data on the page "Image
            Classifier" of this app.
            </li>
            <hr>
            <li>
            The client would like to investigate the possibility of an
            infallible process during which a pet will be either classified
            correctly or not classified at all.
            <br>
            <br>
            ✅ The requirement has been met by conducting hundreds of tests on
            the randomized live data and determining parameters which, when
            implemented, will significantly reduce the risk of a pet
            misclassification.<br>
            The results have been summarized on the page "Recommendations".
            </li>
          </ol>
        </div>
        ''', unsafe_allow_html=True
        )

    st.write("---")
