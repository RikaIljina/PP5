import random
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError


def image_feed_generator(buffer, dims, shuffle=False, repeat=False,
                         fix_dims=False):
    """Generator that prepares and yields one image at a time

    Args:
        buffer (list): Uploaded images
        dims (list): Expected image dimensions
        shuffle (bool, optional): Whether to shuffle the images.
            Defaults to False.
        repeat (bool, optional): Whether to repeat images until all trials have
            been run. Defaults to False.
        fix_dims (bool, optional): Fix the dimensions of invalid images.
            For testing with random inputs. Defaults to False.

    Yields:
        numpy array: Image array to be processed by the model
        ImageFile: Image loaded by PIL to show in the image reel
        list: A collection of errors due to invalid images
    """

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
    """Generator that yields one column at a time

    Yields:
        DeltaGenerator: Next column in a cyclic list of 5
    """
    get_col = iter(st.columns(5))
    while True:
        try:
            col = next(get_col)
        except StopIteration:
            get_col = iter(st.columns(5))
            col = next(get_col)
        yield col
