import streamlit as st
import numpy as np

from src.generators import image_feed_generator, column_generator
from src.utils import get_constants
from src.machine_learning.model_manager import ModelManager


def run_classification(
    trial_amount, min_attempts, max_attempts, min_confidence, images_buffer,
    shuffle, repeat
):
    """Load model and classify images from buffer
    
    This function initializes the generator image_feed_generator and receives
    one image at a time, which is then processed by the model.
    
    To use the keras version instead of tflite:
        model = ModelManager("keras")
        model.load("outputs/x_b_1/model_final_x_b_1.keras")
        
    The generator column_generator yields the next available column from a list
    of 5 columns and is reset whenever a trial ends.

    Args:
        trial_amount (int): Number of trials to run
        min_attempts (int): Mininmum number of images to classify as a batch
        max_attempts (int): Number of images after which the trial is cancelled
        min_confidence (int): Confidence threshold to reach
        images_buffer (list): List of uploaded images
        shuffle (bool): Whether images may be processed in a random order
        repeat (bool): Whether the same images shall be fed to the model until
            the trial_amount value is reached

    Returns:
        dict: Dictionary with the trial results
        list: List of all classes predicted during the run
    """

    model = ModelManager("tflite")
    model.load("outputs/x_b_1/model_x_b_1.tflite")

    CLASS_DICT, LABELS, DIMS = get_constants()

    st.write(
        f"🟢 Model loaded.\n\nConfidence threshold set at "
        f"**{min_confidence}**.\n\nStarting classification ..."
    )
    st.write("---")

    gen_exhausted = False

    feed = image_feed_generator(
        images_buffer, DIMS, shuffle, repeat, fix_dims=True)

    img_nr = 0
    result = {}
    fav_class_min = []

    for t in range(1, trial_amount + 1):
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
                    f"\n❌ Cancelling classification attempt after "
                    f"{max_attempts} attempts.\n\n"
                    f"🔶 Confidence: **{probas_mean[fav_class]:.2f}**, "
                    f"tending towards class: **{CLASS_DICT[fav_class]}**\n"
                )
                st.write("---")
                result[t] = [
                    f"Cancelled after {max_attempts}",
                    f"{f'{CLASS_DICT[fav_class]} ?':<13}",
                    np.around(probas_mean[fav_class], 2),
                ]
                break

            try:
                img_arr, img, errors = next(feed)
            except StopIteration:
                gen_exhausted = True
                wrap_up(
                    min_confidence,
                    CLASS_DICT,
                    attempt_count,
                    errors,
                    probas_mean,
                    fav_class,
                )
                break

            output_data = model.predict(img_arr)

            fav_class_sg = np.argmax(output_data)
            fav_class_min.append(CLASS_DICT[fav_class_sg])
            trial_probas.append(output_data)
            probas_sg = output_data.flatten()
            probas_mean = np.mean(trial_probas, axis=0).flatten()
            fav_class = np.argmax(probas_mean)

            if (
                np.max(probas_mean) < min_confidence
                and attempt_count == min_attempts + 1
            ):
                col = next(get_column)
                con = col.container(border=True)
                unsure = """
                    <span style='text-align: center;'>❓️</span>
                    <br><i>
                    Inconclusive results after analyzing the minimum amount of
                    images, adding images to the batch ...</i>
                    """
                con.markdown(
                    f"""<p class="small-font">{unsure}</p>""",
                    unsafe_allow_html=True,
                )

            col = next(get_column)
            con = col.container(border=False)
            con.markdown(
                f'<p class="small-font">⏩️ Trial <b>{t}</b>, attempt '
                f'{attempt_count}</p>',
                unsafe_allow_html=True,
            )

            if fav_class_sg != np.argmax(probas_mean):
                con.markdown(
                    """<div class="red-bar"></div>
                                """,
                    unsafe_allow_html=True,
                )
            else:
                con.markdown(
                    """<div class="green-bar"></div>
                                """,
                    unsafe_allow_html=True,
                )

            span_green = '<span class="green-bg">'
            span = "<span>"
            span_yellow = '<span class="yellow-bg">'

            proba_str = "<br>".join(
                [
                    ": ".join(
                        [
                            CLASS_DICT[i],
                            f"{span_green if i == fav_class else span}\
                                {probas_mean[i]:.2f}</span>",
                        ]
                    )
                    for i in range(len(probas_mean))
                ]
            )

            proba_sg_str = "<br>".join(
                [
                    ": ".join([c[0], f"{c[1]:.2f}"])
                    for c in zip(LABELS, probas_sg)
                ]
            )
            img_count_annot = (
                f'<p class="tiny-font">{img_nr} of {len(images_buffer)}</p>'
            )

            con.markdown(
                f"""<p class="padding-0-small">{proba_sg_str}
                    <br>Prediction: 
                    {span_yellow}{CLASS_DICT[fav_class_sg].upper()}</span>?
                    </p>
                """, unsafe_allow_html=True,
            )

            con.image(img, use_column_width=True, caption=f"")
            con.markdown(
                f"""{img_count_annot}
                    <p class="padding-0-small">
                    Mean confidence:<br>{proba_str}</p>
                """, unsafe_allow_html=True,
            )

            if attempt_count >= min_attempts:
                if probas_mean[fav_class] > min_confidence / 100:
                    result[t] = [
                        attempt_count,
                        CLASS_DICT[fav_class],
                        np.around(probas_mean[fav_class], 2),
                    ]
                    st.write(
                        f"""
                        ✅ Classification complete:\n\n
                        ✅ Predicted class: 
                        **{CLASS_DICT[fav_class].upper()}**, Confidence: 
                        {probas_mean[fav_class]:.2f}\n\n"""
                    )
                    st.write("---")
                    break

    return result, fav_class_min


def wrap_up(
    min_confidence,
    CLASS_DICT,
    attempt_count,
    errors,
    probas_mean=None,
    fav_class=None,
):
    if attempt_count == 1:
        st.write("The image stream has ended.")
        if len(errors):
            st.write("Errors: ", errors)
    else:
        min_conf_str = (
            f"the minimal confidence of {min_confidence}% hasn't been reached."
        )
        min_amount_str = (f"not enough images have been analyzed to reach a "
                         f"confident conclusion.")
        st.write(
            f"⚠️ The image stream has ended after {attempt_count-1} attempts "
            f"but {min_conf_str if min_confidence/100 > probas_mean[fav_class]\
                else min_amount_str}\n\n"
            f"🔶 Last recorded confidence: {probas_mean[fav_class]:.2f}, "
            f"tending towards class: {CLASS_DICT[fav_class]}\n"
        )
        st.write("---")
