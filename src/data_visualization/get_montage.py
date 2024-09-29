
import os
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from PIL import Image
import itertools
import random
import streamlit as st

#sns.set_style("white")


def set_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


# def show_progress(label, list_len):
#     width = 100
#     for i in range(list_len):
#         yield f'\r{label:<10}: {"#"*int(width if i == list_len-1 else i//(list_len/width)):<{width}}|| '
        



def image_montage(dir_path, label_to_display, nrows, ncols, show_all=False, figsize=(10,10)):
    #sns.set_style("white")
    
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
            img = np.asarray(Image.open(os.path.join(label_path, img))) # / 255.0
            axes[plot_idx[idx][0], plot_idx[idx][1]].imshow(img)
            set_ticks(axes[plot_idx[idx][0], plot_idx[idx][1]])

        plt.suptitle(t=f"\n{label.upper()}:\n", weight="bold", size=20, y=0.85, va="bottom")
        plt.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig=fig)
        
    
