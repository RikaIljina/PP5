import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
from src.data_management import load_pkl_file


# Generator either produces random images from one specific label
# or random images from a random label
def image_feed(buffer, dims):
    errors = [] 
    for img in buffer:
        try:
            img = Image.open(img)
        except (UnidentifiedImageError, IOError) as e:
            errors.append(f"{e} :: {img.name} >> skipped")
            continue
        if img.size != dims:
            img = img.resize(dims, resample=Image.LANCZOS)
        #img_resized = image.img_to_array(img)
        img_arr = np.array(img)
       
        # if len(img_arr.shape) == 3 and img_arr.shape[-1] == 4:
        #     img_arr = img_arr[:, :, 3]
        
        # if len(img_arr.shape) == 2:
        #     img_arr = np.array(np.stack((img_arr, img_arr, img_arr))).reshape(dims[0], dims[1], 3)

        if img_arr.max() > 1:
            img_arr = img_arr / 255
        img_arr = np.expand_dims(img_arr, axis=0)
        
        yield img_arr, img, errors


def get_next_column():
            get_col = iter(st.columns(5))
            while True:
                try:
                    col = next(get_col)
                except StopIteration:
                    get_col = iter(st.columns(5))
                    col = next(get_col)
                yield col
        
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