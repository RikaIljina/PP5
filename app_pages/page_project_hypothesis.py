import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"The goal was to ... \n" 

    )
    
    # st.markdown('''<div class="blue-div">
    #             <h5>Business requirement 4:</h5>
    #             <p>
    #             The client is interested in an assessment regarding the automation of the
    #                 data collection and model training processes.
    #             </p>
    #             </div>
    #         ''', unsafe_allow_html=True)

    st.write("### Have the Business Requirements been met?")
    st.markdown('''<div class="blue-div">
                <ol>
                    <li>
                    The client is interested in a recommendation regarding the scale and quality
                    of future datasets.<br>
                    ... see dataset assessment page
                    </li>
                    <li>
                    The client is interested in a confident and correct classification of any 
                    given live image.<br>
                    ... see image classification page
                    </li>
                    <li>
                    The client is interested in a prototype for a tool that receives and evaluates
                    a stream of snapshots from a camera and returns a useable classification.<br>
                    ... see image classification page
                    </li>
                    <li>
                    The client is interested in an assessment regarding the automation of the
                    data collection and model training processes: <br>
                    ... Need for a new study to find a proper correlation between model 
                    classification results and the histogram comparison metrics for
                    various pet types, sizes and colors. <br>
                    ... Need to find out which features in each label set contribute to
                    which metric going up or down<br>
                    ... Need to find a formula that will analyze the metrics collected
                    during histogram comparison and suggest to the user setting up their
                    device in what ways they can optimize the dataset before training 
                    the model.
                    </li>
                </ol></div>''', unsafe_allow_html=True
        )

    st.write("---")