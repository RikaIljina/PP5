import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

import itertools
import random

def page_dataset_assessment_body():
    output_path = os.path.relpath('outputs')
    full_dataset_path = os.path.relpath('inputs/datasets/pets')
    
    st.write("### Compare pets")
    st.info(
        '''The client is interested in a recommendation based on the comparison of the 
            image labels in the dataset to be able to evaluate and adjust future training
            sets to work with our model.''')
    
    #version = 'v1'
    if st.checkbox("Average and variance images for each label:"):
      
        avg_grid = plt.imread(os.path.join(output_path, 'average_images_grid.png'))
        st.image(avg_grid, caption="The average images for each label in the 'train' subset")
        
        var_grid = plt.imread(os.path.join(output_path, 'variance_images_grid.png'))
        st.image(var_grid, caption="The variance images for each label in the 'train' subset")

        st.success(
        f"* We notice that the average image for each pet is clearly distinguishable "
        f"from the others.\n\n"
        f"* We also notice that the 'iris' label might contain too many similar images, "
        f"seen as clear outlines on the average image. This might lead to bias during "
        f"model training.\n\n"
        f"* The variance images show the mean variance between each image in the label "
        f"set, brighter areas indicating greater variance."
        )

        st.write("---")

    if st.checkbox("Differences between the average images for each label combo:"):
        for combo in ['fin_iris', 'fin_smilla', 'iris_smilla']:
            diff_between_avgs = plt.imread(f"{output_path}/average_imgs_{combo}.png")
            st.image(diff_between_avgs)

        st.success(
        f"* The differences between the average images show by how much each pixel in "
        f"one mean image differs from the same pixel in the other mean image. "
        f"Bright areas indicate greater differences. \n\n"
        f"* We notice that there are visible differences between all labels, with the "
        f"**'fin - smilla'** comparison showing the largest bright area. \n\n"
        f"* We make the preliminary conclusion that the labels **'fin'** and **'smilla'**"
        f" might turn out to be easiest for the model to distinguish."
        )
        
    if st.checkbox("Histogram comparison"):
        st.write(f"Histograms")

    if st.checkbox("Image Montage"):
        st.write("* To refresh the montage, click on the 'Create Montage' button")
        labels = os.listdir(f'{full_dataset_path}/train')
        label_to_display = st.selectbox(label="Select label", options=labels, index=0)
        show_all = st.checkbox("Show all")
        if st.button("Create Montage"):
            image_montage(f'{full_dataset_path}/train', label_to_display,  
                            3, 3, show_all, figsize=(10,10))
        st.write("---")
        

def set_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])




##########

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.preprocessing import image
from PIL import Image
import random

sns.set_style("white")

def show_progress(label, list_len):
    width = 100
    for i in range(list_len):
        yield f'\r{label:<10}: {"#"*int(width if i == list_len-1 else i//(list_len/width)):<{width}}|| '
        
# Load images from specified folder, resize them, save them as np array in X and save their labels in y

from PIL.Image import UnidentifiedImageError

def save_live_data(path, shape):
    Xy_dict = {}
    dims = shape[1:3]
    for label in os.listdir(path):
        img_list = os.listdir(os.path.join(path, label))[:100] # FIXED AMOUNT!!!
        Xy_dict[label] = np.array([], dtype="int")

        print(f'\nImporting from label "{label}..."')
        progress_bar = show_progress(label, len(img_list))
        img_counter = 0
        errors = []

        for img_name in img_list:
            try:
                img = image.load_img(os.path.join(path, label, img_name))
            except UnidentifiedImageError as e:
                errors.append(f"{e} :: {img_name} >> skipped")
                continue
            
            #img = img.crop((30, 30, 90, 90))
            if img.size != (dims):
                img = img.resize(dims, resample=Image.LANCZOS)
            
            img_resized = image.img_to_array(img)
            
            if img_resized.max() > 1:
                img_resized = img_resized / 255

            img_counter += 1
            Xy_dict[label] = np.append(Xy_dict[label], img_resized)

            print(next(progress_bar), end="")

        # Exhaust the generator
        try:
            rest = [p for p in progress_bar]
            print("".join(rest), end="")
        except StopIteration:
            pass

        print(f"{img_counter} images loaded")
        if errors:
            print_err("\n".join(errors))

        Xy_dict[label] = Xy_dict[label].reshape(shape)

    print("\nLive data loaded.")

    return Xy_dict

live_path = os.path.join(full_dataset_path, "train")
X_y_live = save_live_data(live_path, INPUT_SHAPE)

# Check if the images have been loaded correctly
rnd_label = random.choice(list(X_y_live.keys()))
print(f"\nShape of label {'fin'}: {X_y_live.get('fin').shape}")
plt.imshow(random.choice(X_y_live.get('fin')))
plt.axis("off")
plt.show()


# Create a dictionary with the mean images of all labels
import itertools
import functools
from skimage.color import rgb2gray

def get_means(X, y, labels):

    compound_dict = {"means": {}, "vars": {}, "std": {}}

    for label in labels:
        #y = y.reshape(-1, 1, 1)
        #bool_mask = np.any(y == label, axis=1).reshape(-1)
        #arr = X[bool_mask]
        X_mean = np.mean(X[label], axis=0)
        X_var = rgb2gray(np.var(X[label], axis=0))
        compound_dict["means"][label] = X_mean
        compound_dict["vars"][label] = X_var

    return compound_dict


##########################




def image_montage(dir_path, label_to_display, nrows, ncols, show_all=False, figsize=(10,10)):
    sns.set_style("white")
    
    labels = os.listdir(dir_path)
    print(dir_path)
    if show_all:
        st.write('Showing all labels')
    else:
        if label_to_display in labels:
            labels = [label_to_display]
        else:
            print("The label you selected doesn't exist.")
            print(f"The existing options are: {labels}")
            return

    list_rows = range(0, nrows)
    list_cols = range(0, ncols)
    plot_idx = list(itertools.product(list_rows, list_cols))
        
    # subset the class you are interested to display
    for label in labels:
        label_path = os.path.join(dir_path, label)
        images_list = os.listdir(label_path)
    
        if nrows * ncols < len(images_list):
            rnd_sample = random.sample(images_list, nrows * ncols)
        else:
            print(
                f"Decrease nrows or ncols to create your montage. \n"
                f"There are {len(images_list)} in your subset {label}. "
                f"You requested a montage with {nrows * ncols} spaces.")
            return

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 5)
        )
        for idx, img in enumerate(rnd_sample):
            img = np.asarray(Image.open(os.path.join(label_path, img))) / 255.0
            axes[plot_idx[idx][0], plot_idx[idx][1]].imshow(img)
            set_ticks(axes[plot_idx[idx][0], plot_idx[idx][1]])

        plt.suptitle(t=f"\n{label.upper()}:\n", weight="bold", size=20, y=0.85, va="bottom")
        plt.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig=fig)
        
    
