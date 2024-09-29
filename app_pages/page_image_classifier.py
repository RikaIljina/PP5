import streamlit as st
import numpy as np
import pandas as pd
import keras
#from tensorflow.keras.models import load_model
from src.data_management import load_pkl_file
from collections import Counter

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import image_feed, get_next_column


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
        f"image and the comparison of the top value to the confidence threshold.\n\n"
        f"The following parameters may be clipped or adjusted based on the amount "
        f"of provided images.")
    
    st.write("---")
    
    

    if st.session_state.get('tr_input') or st.session_state.get('update_str'):
        update_str = st.session_state['update_str']
    else:
        update_str = ""
        st.session_state['update_str'] = update_str
    
    with st.form(key='param_input_form'):            
        col1, col2, buff = st.columns([6, 3, 1])
        col1.markdown(f'<p style="padding-top: 1rem;">How many consecutive trials should the model run?</p>', unsafe_allow_html=True)
        trial_amount = col2.number_input(label="Amount of trials to run", min_value=1, key='tr_input')
        col1, col2, buff = st.columns([6, 3, 1])
        col1.markdown(f'<p style="padding-top: 1rem;">How many images within one trial should the model analyze before '
                f'trying to reach a conclusion?</p>', unsafe_allow_html=True, help='WARNING: A too low number can lead to misclassification due to a high-confidence error early on')

        min_attempts = col2.number_input(label="Minimal image threshold", min_value=1, value=5)
        col1, col2, buff = st.columns([6, 3, 1])
        col1.markdown(f'<p style="padding-top: 1rem;">After how many images should the model give up the classification '
                f'attempt and restart the trial?</p>', unsafe_allow_html=True)
        max_attempts = col2.number_input(label="Max attempts per trial", min_value=1, value=10)
        col1, col2, buff = st.columns([6, 3, 1])
        col1.markdown(f'<p style="padding-top: 1rem;">What confidence level should count as a successful classification?', unsafe_allow_html=True)
        min_confidence = col2.slider(label="Confidence threshold", min_value=0, max_value=100, value=75)
        
        submitted_params = st.form_submit_button("SAVE")
        
        warning_box = st.empty()
        
        if submitted_params:
            if max_attempts < min_attempts:
                warning_box.error(
                    f"The value for 'Max attempts per trial' must be equal to or higher "
                    f"than the value for 'Minimal image threshold'. Adjusting.")
                max_attempts = min_attempts
            else:
                warning_box.text("")

            batch_range_str = (f"**{min_attempts}**" if min_attempts == max_attempts 
                               else f"**{min_attempts}** - **{max_attempts}**")

            update_str = (
                f"The model will conduct **{trial_amount}** trial{'s' if trial_amount > 1 else ''} with {batch_range_str} "
                f"images each. A trial will end as soon as the model is "
                f"**{min_confidence}%** confident in its prediction."
                )
        
            st.session_state['update_str'] = update_str

    update_box = st.empty()
    update_box.info(update_str)#st.session_state['update_str'])
    
    st.write("---")
    
    start_model = False
    
    keep = not st.checkbox('Keep files in cache')

    with st.form("img-upload-form", clear_on_submit=keep):
        images_buffer = st.file_uploader('Upload pet images. You may select more than one.',
                                    type='png',accept_multiple_files=True)
        submitted_img = st.form_submit_button("START CLASSIFICATION")
        
        if submitted_img and images_buffer is not None:
            start_model = True
    
    print(images_buffer)
    
    if images_buffer and start_model:

        if st.session_state['tr_input']:
            batch_range_str = (f"**{min_attempts}**" if min_attempts == max_attempts 
                               else f"**{min_attempts}** - **{max_attempts}**")
            update_str = (
                f"The model will conduct **{trial_amount}** trial{'s' if trial_amount > 1 else ''} with {batch_range_str} "
                f"images each. A trial will end as soon as the model is "
                f"**{min_confidence}%** confident in its prediction."
                )
            update_box.info(update_str)#st.session_state['update_str'])
        else:
            st.write(st.session_state.items())
            
            
        st.write('Loading the model ...')
        try:
            #model = load_model(f"outputs/model_final.keras")
            model = keras.saving.load_model(f"outputs/model_final.keras")
        except:
            st.write("Error.")

        classes = load_pkl_file(file_path=f'outputs/class_dict.pkl')
        LABELS = sorted(classes.values())
        dims = load_pkl_file(file_path=f"outputs/input_shape.pkl")[1:3]
        st.write(f'🟢 Model loaded.\n\nConfidence threshold set at **{min_confidence}**.\n\nStarting classification ...')
        st.write("---")

        gen_exhausted = False
        # Initialize random generator
        
        feed = image_feed(images_buffer, dims)
        
        img_nr = 0
        result = {}
        # Amount of trials for each label
        #trial_amount = 10
        # Minimum amount of images to be evaluated as one batch; a too low number can
        # lead to misclassification due to a high-confidence error early on
        #min_attempts = min(min_attempts, len(images_buffer))
        # Minimum accepted confidence threshold
        #min_confidence = 75
        # Max amount of attempts after which the current trial is aborted and the next trial started
        max_attempts = min(max_attempts, len(images_buffer))
        fav_class_min = []

        for t in range(1, trial_amount+1):
            if gen_exhausted:
                break
            attempt_count = 0
            trial_probas = []
            
            # col1, col2, col3, col4, col5 = st.columns(1,1,1,1,1)
            # all_cols = [col1, col2, col3, col4, col5]
            get_col = get_next_column()

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
                    st.write("---")
                    result[t] = [f"Cancelled after {max_attempts}", f"{f'{classes[fav_class]} ?':<13}",
                            np.around(probas_mean[fav_class], 2),]
                    break
                
                # Prepare image and use model to predict a class
                try:
                    img_arr, img, errors  = next(feed)
                    # ----------- Uncomment to see a reel of currently evaluated images -------
                    # Predict based on feed-in data
                    # Run [img_min] predictions first and collect data without trying to classify
                    y_pred_sg = model.predict(img_arr, verbose=0)
                    fav_class_sg = np.argmax(y_pred_sg)
                    fav_class_min.append(classes[fav_class_sg])

                    trial_probas.append(y_pred_sg)
                    probas_sg = y_pred_sg.flatten()
                    probas_mean = np.mean(trial_probas, axis=0).flatten()
                    fav_class = np.argmax(probas_mean)
                    
                    if np.max(probas_mean) < min_confidence and attempt_count == min_attempts+1:
                        col = next(get_col)
                        con = col.container(border=True)
                        unsure = "<span style='text-align: center;'>❓️</span><br><i>Inconclusive results after analyzing the minimum amount of images, adding images to the batch ...</i>"
                        con.markdown(f'''<p class="small-font" style="height: 100%; padding-top: 6rem; padding-bottom: 6rem;">{unsure}</p>''', unsafe_allow_html=True)
                    
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

                    hl = {'green': {'open': f'<span style="background-color: #B9F5BB;">', 'close': '</span>'},
                          'yellow': {'open': f'<span style="background-color: #FCF8AE;">', 'close': '</span>'},
                          }
                    proba_str = "<br>".join([": ".join([classes[i], f"{hl['green']['open'] if i == fav_class else ''}{probas_mean[i]:.2f}{hl['green']['close'] if i == fav_class else ''}"]) for i in range(len(probas_mean))])
                    proba_sg_str = "<br>".join([": ".join([c[0], f"{c[1]:.2f}"]) for c in zip(LABELS, probas_sg)])
                    img_count_annot = f'<p style="padding-top: 0 !important; font-size: 0.5rem; text-align: right;">{img_nr} of {len(images_buffer)}</p>'
                    
                    con.markdown(f'''<p class="small-font" style="padding-bottom: 0 !important; padding-top: 0 !important;">
                                 {proba_sg_str}<br>Prediction: {hl['yellow']['open']}{classes[fav_class_sg].upper()}{hl['yellow']['close']}?
                                 </p>''', 
                                 unsafe_allow_html=True)
                    
                    con.image(img, use_column_width=True, caption=f"")
                    con.markdown(f'''{img_count_annot}<p class="small-font" style="padding-bottom: 0 !important; padding-top: 0 !important;">
                                 Mean confidence:<br>{proba_str}</p>''', 
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
                        f"⚠️ The image stream has ended after {attempt_count-1} attempts but "
                        f"{min_conf_str if min_confidence/100 > probas_mean[fav_class] else min_amount_str}\n\n"
                        f"🔶 Last recorded confidence: {probas_mean[fav_class]:.2f}, tending towards class: {classes[fav_class]}\n")
                    st.write("---")

                    break

                if attempt_count >= min_attempts:
                    if probas_mean[fav_class] > min_confidence / 100:
                        result[t] = [attempt_count, classes[fav_class], np.around(probas_mean[fav_class], 2)]
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
            st.success("### **Analysis Report**")
            st.write(f'#### Confidence was set to **{min_confidence}%**')
            cap1 = f"Batch classification results after {len(df)} Trials"
            st.write(cap1)
            
            st.dataframe(
                df.style
                .format(
                    {"Confidence": "{:.2f}".format})
                .apply(
                    lambda x: ['background-color: #FCF8AE']*len(x) 
                    if not isinstance(x.loc['Attempts'], int) else ['']*len(x), axis=1))

            df_favs_maj = pd.DataFrame.from_dict(Counter(df["Predicted class"]), columns=["Amount"], orient="index") # pd.DataFrame.from_dict(Counter(fav_class_maj), columns=["Amount"], orient="index")
            df_favs_maj.index.name = 'Class'
            
            df_favs_sg = pd.DataFrame.from_dict(Counter(fav_class_min), columns=["Amount"], orient="index")
            df_favs_sg.index.name = 'Class'
            
            col1, col2 = st.columns(2)
            with col1:
                cap2 = "Sum of the final conclusions made by the model"
                st.caption(cap2)
                st.dataframe(df_favs_maj)
            with col2:
                cap3 = "Sum of individual predictions for each image"
                st.caption(cap3)
                st.dataframe(df_favs_sg)

            st.markdown(download_dataframe_as_csv((df, df_favs_maj, df_favs_sg), (cap1, cap2, cap3)), unsafe_allow_html=True)
        