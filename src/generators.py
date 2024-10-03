import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from src.data_management import load_pkl_file
import random


# Generator either produces random images from one specific label
# or random images from a random label
def image_feed_generator(buffer, dims, shuffle=False, repeat=False, fix_dims=False):
    errors = []
    if shuffle:
        random.shuffle(buffer)

    feed = True if shuffle and repeat else len(buffer)
    while feed:
        if repeat:
            rand_idx = random.randint(0, len(buffer) - 1)
            img = buffer[rand_idx]
        else:
            img = buffer[len(buffer) - feed]
            feed -= 1
        try:
            img = Image.open(img)
        except (UnidentifiedImageError, IOError) as e:
            errors.append(f"{e} :: {img.name} >> skipped")
            continue
        if img.size != dims:
            img = img.resize(dims, resample=Image.LANCZOS)
        # img_resized = image.img_to_array(img)
        img_arr = np.array(img).astype("float32")

        if fix_dims:
            if len(img_arr.shape) == 3 and img_arr.shape[-1] == 4:
                img_arr = img_arr[:, :, 3]
            if len(img_arr.shape) == 2:
                img_arr = np.array(
                    np.stack((img_arr, img_arr, img_arr))
                ).reshape(dims[0], dims[1], 3)

        if img_arr.max() > 1:
            img_arr = img_arr / 255
        img_arr = np.expand_dims(img_arr, axis=0)

        yield img_arr, img, errors


def column_generator():
    get_col = iter(st.columns(5))
    while True:
        try:
            col = next(get_col)
        except StopIteration:
            get_col = iter(st.columns(5))
            col = next(get_col)
        yield col
