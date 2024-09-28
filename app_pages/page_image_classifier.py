import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.data_management import load_pkl_file
import random

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import X_gen
# resize_input_image # load_model_and_predict, plot_predictions_probabilities

def page_image_classifier_body():
    
    st.markdown("""
    <style>
    .small-font {
        font-size:0.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
            <div style="background-color: #CBDFE3; padding: 1rem;">
            <p>The client is interested in classifying pet images from an incoming image stream.
            </p>
            </div>
    """, unsafe_allow_html=True)

    st.write('\n\n')

    st.markdown("""
        <div style="background-color: #E3E2CB; padding: 1rem;">
        <p>
        You can download a set of pet images for live prediction. You can download the 
        images from Github: 
        <a href="https://github.com/RikaIljina/PP5/main/README.md" 
        target="_blank" rel="noopener">GitHub</a>.
        </p>
        </div>
        """ , unsafe_allow_html=True)

    st.write('\n\n')

    st.write("---")
    st.info(f"**Please provide parameters for the classification.**\n"
        f"* A **trial** is a series of predictions from individual images that are assumed "
        f"to belong to the same class. The final classification result is based on "
        f"the mean values of all classification attempts within that trial.\n"
        f"* An **attempt** is the model's calculation of probabilities for a single "
        f"image and the comparison of the top value to the confidence threshold."
        f"The following parameters may be clipped or adjusted based on the amount "
        f"of the provided images.")
    
    st.write("---")
    
    col1, col2, buff = st.columns([6, 3, 1])
    col1.markdown(f'<p style="padding-top: 1rem;">How many consecutive trials should the model run?</p>', unsafe_allow_html=True)
    trial_amount = col2.number_input(label="Amount of trials to run", min_value=1)
    col1, col2, buff = st.columns([6, 3, 1])
    col1.markdown(f'<p style="padding-top: 1rem;">How many images within one trial should the model analyze before '
               f'attempting a classification?</p>', unsafe_allow_html=True, help='WARNING: A too low number can lead to misclassification due to a high-confidence error early on')

    img_min = col2.number_input(label="Minimal image threshold", min_value=1, value=5)
    col1, col2, buff = st.columns([6, 3, 1])
    col1.markdown(f'<p style="padding-top: 1rem;">After how many images should the model give up the classification '
               f'attempt and restart the trial?</p>', unsafe_allow_html=True)
    max_attempts = col2.number_input(label="Max attempts per trial", min_value=img_min, value=10)
    col1, col2, buff = st.columns([6, 3, 1])
    col1.markdown(f'<p style="padding-top: 1rem;">What confidence level should count as a successful classification?', unsafe_allow_html=True)
    min_confidence = col2.slider(label="Confidence threshold", min_value=0, max_value=100, value=75)
    
    st.write("---")
    
    start_model = False
    
    with st.form("img-upload-form", clear_on_submit=True):
        images_buffer = st.file_uploader('Upload pet images. You may select more than one.',
                                    type='png',accept_multiple_files=True)
        submitted = st.form_submit_button("START CLASSIFICATION")
        
        if submitted and images_buffer is not None:
            start_model = True
    
    
    print(images_buffer)
    if images_buffer and start_model: # is not None:
        
        def get_next_col():
            # col1, col2, col3, col4, col5 = st.columns(5)
            # all_cols = [col1, col2, col3, col4, col5]
            get_col = iter(st.columns(5))
            while True:
                try:
                    col = next(get_col)
                except StopIteration:
                    get_col = iter(st.columns(5))
                    col = next(get_col)
                yield col
            
        st.write('Loading the model ...')
        try:
            model = load_model(f"outputs/model_final.keras")
        except:
            st.write("Error.")
            
        classes = load_pkl_file(file_path=f'outputs/class_dict.pkl')
        LABELS = sorted(classes.values())
        dims = load_pkl_file(file_path=f"outputs/input_shape.pkl")[1:3]
        st.write(f'🟢 Model loaded.\n\nConfidence threshold set at **{min_confidence}**.\n\nStarting classification ...')
        st.write("---")

        gen_exhausted = False
        # Initialize random generator
        
        xgen = X_gen(images_buffer, dims)
        
        img_nr = 0
        result = {}
        # Amount of trials for each label
        #trial_amount = 10
        # Minimum amount of images to be evaluated as one batch; a too low number can
        # lead to misclassification due to a high-confidence error early on
        img_min = min(img_min, len(images_buffer))
        # Minimum accepted confidence threshold
        #min_confidence = 75
        # Max amount of attempts after which the current trial is aborted and the next trial started
        max_attempts = min(max_attempts, len(images_buffer))

        for t in range(1, trial_amount+1):
            if gen_exhausted:
                break
            attempt_count = 0
            trial_probas = []
            
            # col1, col2, col3, col4, col5 = st.columns(1,1,1,1,1)
            # all_cols = [col1, col2, col3, col4, col5]
            get_col = get_next_col()

            # Run trial until the desired confidence threshold has been reached
            # or until the attempt count has reached max attempts.
            
            while True:
                attempt_count += 1
                img_nr += 1
                if attempt_count > max_attempts:
                    st.write(
                        f'\n❌ Cancelling classification attempt after {max_attempts} attempts.\n\n'
                        f'🔶 Confidence: **{probas_mean[fav_class]:.2f}**, tending towards class: **{classes[fav_class]}**\n'
                    )
                    result[t] = ["❌", f"{f'{classes[fav_class]} ❓️':<13}",
                            probas_mean[fav_class],]
                    break
                
                # Prepare image and use model to predict a class
                try:
                    img_arr, img, errors  = next(xgen)
                    # ----------- Uncomment to see a reel of currently evaluated images -------
                    # Predict based on feed-in data
                    # Run [img_min] predictions first and collect data without trying to classify
                    y_pred_sg = model.predict(img_arr, verbose=0)
                    fav_class_sg = np.argmax(y_pred_sg)

                    trial_probas.append(y_pred_sg)
                    probas_sg = y_pred_sg.flatten()
                    probas_mean = np.mean(trial_probas, axis=0).flatten()
                    fav_class = np.argmax(probas_mean)
                    
                    if np.max(probas_mean) < min_confidence and attempt_count == img_min+1:
                        col = next(get_col)
                        con = col.container(border=True)
                        unsure = "❓️<br>Inconclusive results after analyzing the minimum amount of images, adding images to the batch ..."
                        con.markdown(f'''<p class="small-font" style="height: 100%; padding-top: 1rem;">{unsure}</p>''', unsafe_allow_html=True)
                    
                    col = next(get_col)
                    con = col.container(border=False)
                    con.markdown(f'<p class="small-font">⏩️ Trial <b>{t}</b>, attempt {attempt_count}</p>', unsafe_allow_html=True)
                    st.write()
                    if fav_class_sg != np.argmax(probas_mean):
                        con.markdown('''<div style="background-color: red; width: 100%; height: 2px;"></div>
                                    ''', unsafe_allow_html=True)
                    else:
                        con.markdown('''<div style="background-color: green; width: 100%; height: 2px;"></div>
                                    ''', unsafe_allow_html=True)

                    hl = [f'<span style="background-color: yellow;">', '</span>']
                    proba_str = "<br>".join([": ".join([classes[i], f"{hl[0] if i == fav_class else ''}{probas_mean[i]:.2f}{hl[1] if i == fav_class else ''}"]) for i in range(len(probas_mean))])
                    #proba_str = "<br>".join([": ".join([c[0], f"{c[1]:.2f}"]) for c in zip(LABELS, probas_mean)])
                    proba_sg_str = "<br>".join([": ".join([c[0], f"{c[1]:.2f}"]) for c in zip(LABELS, probas_sg)])
                    img_count_annot = f'<p style="padding-top: 0 !important; font-size: 0.5rem; text-align: right;">{img_nr} of {len(images_buffer)}</p>'
                    
                    con.markdown(f'''<p class="small-font" style="padding-bottom: 0 !important; padding-top: 0 !important;">
                                 Is this {hl[0]}{classes[fav_class_sg].upper()}{hl[1]}?
                                 <br>{proba_sg_str}</p>{img_count_annot}''', 
                                 unsafe_allow_html=True)
                    
                    con.image(img, use_column_width=True, caption=f"")
                    con.markdown(f'''<p class="small-font" style="padding-bottom: 0 !important; padding-top: 0 !important;">
                                 Mean confidence:<br>{proba_str}</p>{img_count_annot}''', 
                                 unsafe_allow_html=True)

                    # -------------------------------------------------------------------------
                    # Keep track of all probabilities in the current trial

                except StopIteration:
                    gen_exhausted = True
                    if attempt_count == 1:
                        st.write("The image stream has ended.")
                        st.write("Errors: ", errors)
                        break
                    
                if gen_exhausted:
                    min_conf_str = f"the minimal confidence of {min_confidence}% hasn't been reached."
                    min_amount_str = f"not enough images have been analyzed to reach a confident conclusion."
                    st.write(
                        f"⚠️ The image stream has ended after {attempt_count} attempts but "
                        f"{min_conf_str if min_confidence/100 > probas_mean[fav_class] else min_amount_str}\n\n"
                        f"🔶 Last recorded confidence: {probas_mean[fav_class]:.2f}, tending towards class: {classes[fav_class]}\n")
                    break

                if attempt_count >= img_min:
                    if probas_mean[fav_class] > min_confidence / 100:
                        result[t] = [attempt_count, classes[fav_class], probas_mean[fav_class]]
                        st.write(
                        f'✅ Classification complete:\n\n'
                        f'✅ Predicted class: **{classes[fav_class].upper()}**, Confidence: {probas_mean[fav_class]:.2f}\n\n'
                        )
                        st.write("---")
                        break
                    # else:
                    #     col.markdown(f'<p class="small-font" style="height:70px; padding-bottom: 0 !important; padding-top: 0 !important;">❓️ Unsure yet', unsafe_allow_html=True)
 
        column_list = ["Attempts", "Predicted class", "Confidence"]
        df = pd.DataFrame.from_dict(result, orient="index", columns=column_list)
        df.index.name = 'Trial nr'

        if not df.empty:
            st.success("Analysis Report")
            st.write(f'Confidence was set to {min_confidence}')
            st.markdown(df.to_html(), unsafe_allow_html=True)
            #st.table(df)
            st.markdown(download_dataframe_as_csv(df), unsafe_allow_html=True)
            
        '''
        
        df_report = pd.DataFrame([])
        
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Pet pic: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            #plot_predictions_probabilities(pred_proba, pred_class)

            #df_report = df_report.append({"Name":image.name, 'Result': pred_class },
            #                            ignore_index=True)
            st.write(pred_class)
        
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)
'''

