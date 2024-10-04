import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Project Summary")

    st.markdown("""
            <div class="blue-div">
                <h5>General Information</h5>
                <p>
                For many people, pets are an important part of their lives. However, some people 
                struggle to keep track of all their pets' needs due to an irregular work schedule 
                or their own health issues.
                </p>
                <p>
                Moreover, <b>pet obesity</b> has increased dramatically over the past decade, with up to
                60% of all pets meeting the criteria of being obese, according to a 
                <a href="https://veterinairepetcare.com/blog/key-pet-obesity-statistics-facts-2023.html" 
                target="_blank" rel="noopener">study by APOP</a>.
                <br><br>
                One of the reasons for pet obesity is overfeeding.
                <br><br>
                To lighten the pet owners' burden while ensuring the pets' wellbeing,
                the client is building an <b>automated food dispenser</b> for pet animals. The product
                is targeted towards owners with at least two visually distinguishable animals.
                </p>
                <p>
                The product will ensure that each pet that comes up to the feeder receives the
                correct type and amount of food at preset intervals, allowing the pet owner to adjust the
                pets' diets in accordance to the veterinarian's advice.
                </p>
                <p>
                To this end, the product will be equipped with a motion sensor and a camera that will 
                take a series of snapshots whenever it detects movement. The camera will take
                three pictures per second. The device software will then run the image batch
                against a pretrained model, deciding which pet has triggered its activation.
                </p>
                <p>
                In case of a successful classification, it will proceed to dispense food and record the 
                interaction for further assessment.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.write("\n\n")
    
    st.markdown("""
            <div class="yellow-div">
            <p>
                For additional information, please visit the 
                <a href="https://github.com/RikaIljina/PP5/main/README.md" 
                target="_blank" rel="noopener">Project README file on GitHub</a>.
            </p>
            <p>
                Please download the full dataset and the live image set from Google Drive:
                <br>
                📸 <a href="https://drive.usercontent.google.com/u/0/uc?id=1M4vruKofgkxSTwYd1FCdtoajfvmZFP6Y&export=download" 
                target="_blank" rel="noopener">Live images</a>
                <br>
                💾 <a href="https://drive.usercontent.google.com/download?id=1jDeB3UaS86FiSKJnJ4Q5-n01PZ2Fq-zr&export=download" 
                target="_blank" rel="noopener">Train, Test, Validation datasets</a>
            </p>
            </div>
            """ , unsafe_allow_html=True)
    # st.write(
    #     f"* For additional information, please visit and **read** the "
    #     f"[Project README file](https://github.com/RikaIljina/PP5/main/README.md).")
    
    st.write("\n\n")

    st.markdown("""
        <div class="blue-div">
          <b>The project has 3 business requirements:</b>
          <ol>
            <li>
            The client is interested in a recommendation regarding the scale 
            and quality of future datasets as well as an investigation of a 
            correlation between the similarity of the pets' visual features and
            the performance of the model.
            </li>
            <li>
            The client is interested in a proof-of-concept model that will tell
            pets apart by their images and achieve an F1 score > 0.9 for each
            label. 
            </li>
            <li>
            The client would like to investigate the possibility of an 
            infallible process during which a pet will be either classified 
            correctly or not classified at all.
            </li>
          </ol>
        </div>
        """ , unsafe_allow_html=True)
