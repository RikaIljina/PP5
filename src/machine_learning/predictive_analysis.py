import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from src.data_management import load_pkl_file
#from src.machine_learning.predictive_analysis import load_model, get_constants
from src.generators import image_feed_generator, column_generator


class ModelManager():
    def __init__(self, type):
        self.type = type

    def load(self, path):
        if self.type == "keras":
            self.model = load_model(path)  # "outputs/model_final.keras"
            return self.model
        
        if self.type == "tflite":
            self.interpreter = tf.lite.Interpreter(model_path=path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def predict(self, img_object):
        if self.type == "keras":
            return self.model.predict(img_object)
        
        if self.type == "tflite":
            input_index = self.input_details[0]['index']
            output_index = self.output_details[0]['index']
            self.interpreter.set_tensor(input_index, img_object)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_index)
            return output_data
        


def get_constants():
    CLASS_DICT = load_pkl_file(file_path=f'outputs/class_dict.pkl')
    LABELS = sorted(CLASS_DICT.values())
    DIMS = load_pkl_file(file_path=f"outputs/input_shape.pkl")[1:3]

    return CLASS_DICT, LABELS, DIMS


def run_classification(trial_amount, min_attempts, max_attempts, min_confidence, images_buffer):
    
    model = ModelManager('tflite')
    model.load("outputs/model.tflite")
    
    CLASS_DICT, LABELS, DIMS = get_constants()
    
    st.write(f'🟢 Model loaded.\n\nConfidence threshold set at **{min_confidence}**.\n\nStarting classification ...')
    st.write("---")

    gen_exhausted = False

    feed = image_feed_generator(images_buffer, DIMS)

    img_nr = 0
    result = {}
    max_attempts = min(max_attempts, len(images_buffer))
    fav_class_min = []

    for t in range(1, trial_amount+1):
        if gen_exhausted:
            break
        attempt_count = 0
        trial_probas = []
        probas_mean = []
        fav_class = None
            
        get_column = column_generator()

            # Run trial until the desired confidence threshold has been reached
            # or until the attempt count has reached max attempts.
            
        while True:
            attempt_count += 1
            img_nr += 1
            if attempt_count > max_attempts:
                st.write(
                        f'\n❌ Cancelling classification attempt after {max_attempts} attempts.\n\n'
                        f'🔶 Confidence: **{probas_mean[fav_class]:.2f}**, tending towards class: **{CLASS_DICT[fav_class]}**\n'
                    )
                st.write("---")
                result[t] = [f"Cancelled after {max_attempts}", f"{f'{CLASS_DICT[fav_class]} ?':<13}",
                            np.around(probas_mean[fav_class], 2),]
                break
                
            # Prepare image and use model to predict a class
            try:
                img_arr, img, errors  = next(feed)
            except StopIteration:
                gen_exhausted = True
                wrap_up(min_confidence, CLASS_DICT, attempt_count, errors, probas_mean, fav_class)
                break
                
            output_data = model.predict(img_arr)
            
                   #y_pred_sg = model.predict(img_arr, verbose=0)
                    # if len(y_pred_raw) == 0:
                    #     y_pred_raw = output_data
                    # else:
                    #     y_pred_raw = np.vstack((y_pred_raw, output_data))
                #fav_class_sg = np.argmax(y_pred_sg)
                    

            fav_class_sg = np.argmax(output_data)
            fav_class_min.append(CLASS_DICT[fav_class_sg])
            trial_probas.append(output_data)
            probas_sg = output_data.flatten()
            probas_mean = np.mean(trial_probas, axis=0).flatten()
            fav_class = np.argmax(probas_mean)
                
            if np.max(probas_mean) < min_confidence and attempt_count == min_attempts+1:
                col = next(get_column)
                con = col.container(border=True)
                unsure = '''
                    <span style='text-align: center;'>❓️</span>
                    <br><i>
                    Inconclusive results after analyzing the minimum amount of images,
                    adding images to the batch ...</i>
                    '''
                con.markdown(f'''<p class="small-font">{unsure}</p>''', unsafe_allow_html=True)
                
            col = next(get_column)
            con = col.container(border=False)
            con.markdown(f'<p class="small-font">⏩️ Trial <b>{t}</b>, attempt {attempt_count}</p>', unsafe_allow_html=True)

            if fav_class_sg != np.argmax(probas_mean):
                con.markdown('''<div class="red-bar"></div>
                                ''', unsafe_allow_html=True)
            else:
                con.markdown('''<div class="green-bar"></div>
                                ''', unsafe_allow_html=True)

            span_green = '<span class="green-bg">'
            span = '<span>'
            span_yellow = '<span class="yellow-bg">'
            
            proba_str = "<br>".join(
                [": ".join([CLASS_DICT[i], 
                            f"{span_green if i == fav_class else span}{probas_mean[i]:.2f}</span>"]) for i in range(len(probas_mean))])
            
            proba_sg_str = "<br>".join([": ".join([c[0], f"{c[1]:.2f}"]) for c in zip(LABELS, probas_sg)])
            img_count_annot = f'<p class="tiny-font">{img_nr} of {len(images_buffer)}</p>'
                
            con.markdown(f'''<p class="padding-0-small">
                                {proba_sg_str}<br>Prediction: {span_yellow}{CLASS_DICT[fav_class_sg].upper()}</span>?
                                </p>''', 
                                unsafe_allow_html=True)

            con.image(img, use_column_width=True, caption=f"")
            con.markdown(f'''{img_count_annot}<p class="padding-0-small">
                                Mean confidence:<br>{proba_str}</p>''', 
                                unsafe_allow_html=True)

                # -------------------------------------------------------------------------
                # Keep track of all probabilities in the current trial

        

            if attempt_count >= min_attempts:
                if probas_mean[fav_class] > min_confidence / 100:
                    result[t] = [attempt_count, CLASS_DICT[fav_class], np.around(probas_mean[fav_class], 2)]
                    st.write(
                        f'✅ Classification complete:\n\n'
                        f'✅ Predicted class: **{CLASS_DICT[fav_class].upper()}**, Confidence: {probas_mean[fav_class]:.2f}\n\n'
                        )
                    st.write("---")
                    break

    return result, fav_class_min


def wrap_up(min_confidence, CLASS_DICT, attempt_count, errors, probas_mean=None, fav_class=None):
    if attempt_count == 1:
        st.write("The image stream has ended.")
        if len(errors):
            st.write("Errors: ", errors)
    else:
        min_conf_str = f"the minimal confidence of {min_confidence}% hasn't been reached."
        min_amount_str = f"not enough images have been analyzed to reach a confident conclusion."
        st.write(
                            f"⚠️ The image stream has ended after {attempt_count-1} attempts but "
                            f"{min_conf_str if min_confidence/100 > probas_mean[fav_class] else min_amount_str}\n\n"
                            f"🔶 Last recorded confidence: {probas_mean[fav_class]:.2f}, tending towards class: {CLASS_DICT[fav_class]}\n")
        st.write("---")


'''

def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results
    """
    classes = load_pkl_file(file_path=f'outputs/class_dict.pkl')
    prob_per_class = pd.DataFrame(
        data=[0, 0, 0],
        index=classes.keys(),
        columns=['Probability']
    )
    prob_per_class.loc[pred_class] = pred_proba
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y=prob_per_class['Probability'],
        range_y=[0, 1],
        width=600, height=300, template='seaborn')
    st.plotly_chart(fig)


def resize_input_image(img, shape):
    """
    Reshape image to average image size
    """
    if img.size != shape:
        img_resized = img.resize(shape, Image.LANCZOS)
    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image


def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """

    model = load_model(f"outputs/model_final.keras")

    pred_proba = model.predict(my_image)[0, 0]

    target_map = {v: k for k, v in {'Fin': 0, 'Iris': 1, 'Smilla': 2}.items()}
    pred_class = target_map[pred_proba > 0.5]
    if pred_class == target_map[0]:
        pred_proba = 1 - pred_proba

    st.write(
        f"The predictive analysis indicates the pet is "
        f"**{pred_class.lower()}** {pred_class}.")

    return pred_proba, pred_class
'''