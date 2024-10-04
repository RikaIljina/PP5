import streamlit as st


def page_recommendations_body():

    st.write(f"#### Conclusions and recommendations for classification "
             f"parameters")

    st.info(
        """
        In order to find adequate classification parameters, tests were 
        conducted at different confidence thresholds.\n\n
        At a confidence threshold of 60%, after 3 images, “smilla” was
        misclassified as “iris” at a probability of 0.6 once during 500 trials.
        \n
        At a confidence threshold of 80%, after 15 images, “smilla” was not
        classified at all due to a mean probability of 0.78 once during 500
        trials.\n
        At a confidence threshold of 70%, after 3-11 images, all pets were
        classified correctly with a mean probability of 0.71 during 500 
        trials.\n\n
        Based on these results, the client receives the following
        recommendation:\n
        """)
    st.success(
        """
        Factoring in an adequate margin of safety, the minimum batch size
        should be at least **5 images**, the maximum batch size **15 images**,
        and the confidence threshold **70%**.\n
        This is deemed sufficient to substantially reduce the risk of
        misclassification while ensuring that the pet with the lowest recall
        value still gets identified within a short amount of time.
        """)

    st.write("#### Prediction of Pair Distinguishability")

    st.write(
        """
        The detailed data on label misclassification was collected during
        hundreds of trials with the best-performing models and is based on the
        358 independent live images of the three pets. The trials were 
        conducted on equally sized batches for all labels. Different models
        were used to exclude the risk of one model favouring one specific 
        label.
        """)

    st.code("""
            '''
            Total count:
    
    Misclassified as |  FIN	    IRIS    SMILLA
    --------------------------------------------
     fin	         |  0	    438	     122
     iris	         |  925	    0	     2016
     smilla	         |  0	    952	     0

    --------------------------------------------

           | Pet was being      | Pet was not
           | wrongly identified | identified 
           | (false positives)  | (false negatives)
    -----------------------------------------------
    Fin              560                 925	
    Iris            2941                1390	
    Smilla           952                2138

    --------------------------------------------
    Pet1   -> mistaken for Pet2  |  times
    --------------------------------------------
    fin    -> smilla             |	0
    smilla -> fin                |  122
    fin    -> iris               |  925
    iris   -> fin                |  438
    iris   -> smilla             |  952
    smilla -> iris               |  2016

    --------------------------------------------
    Summary by pair
    --------------------------------------------
    Fin-Smilla pair  :   122
    Fin-Iris pair    :   1363
    Iris-Smilla pair :   2968

            '''
            """)

    st.image('outputs/misclass_pie.png',
             caption="Misclassification ratio between the pairs")

    st.write("#### Could it have been predicted?")

    st.success(
        """
        The misclassification results are corroborated by the following
        image assessment methods that were performed on the page "Dataset 
        Assessment":\n
        - The Variance values of the images showing the difference between the
        Mean images of each label pair placed the pairs in the same order,
        with a value of 0.01, 0.007 and 0.004 respectively.
        - The values yielded by the Intersection method placed the pairs in the
        same order on the similarity scale, with the distance between 
        'Fin-Smilla' and 'Fin-Iris' smaller than the distance between 
        'Fin-Iris' and 'Iris-Smilla'.
        - The values yielded by the Bhattacharyya distance calculation placed 
        the pairs in the same order on the similarity scale, with a smaller
        difference between 'Fin-Iris' and 'Iris-Smilla' than between either and
        'Fin-Smilla'.
        """)

    st.markdown("""
        An analysis of all methods can be found in the project's <a href=
        "https://github.com/RikaIljina/PP5/blob/main/README.md" 
        target="_blank" rel="noopener">README on GitHub</a>.
        """, unsafe_allow_html=True
                )

    st.markdown(
        """
        In conclusion, an indication for a correlation between certain
        histogram comparison metrics and the model classification scores 
        was found, which provides an answer to <b>Business Requirement 1</b>.
        <br>
        This establishes the need for an additional study where
        the correlation will be researched and calculated. A crucial 
        element for that study is a comprehensive image dataset with 
        various pet types, sizes and colors.<br>
        """, unsafe_allow_html=True
    )
