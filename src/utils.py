import streamlit as st

from data_management import load_pkl_file



def get_constants():
    CLASS_DICT = load_pkl_file(file_path=f"outputs/class_dict.pkl")
    LABELS = sorted(CLASS_DICT.values())
    DIMS = load_pkl_file(file_path=f"outputs/input_shape.pkl")[1:3]

    return CLASS_DICT, LABELS, DIMS


def update_info_box(
    trial_amount, min_attempts, max_attempts, min_confidence, update_box
):
    if st.session_state["tr_input"]:
        batch_range_str = (
            f"**{min_attempts}**"
            if min_attempts == max_attempts
            else f"**{min_attempts}** - **{max_attempts}**"
        )
        update_str = (
            f"The model will conduct **{trial_amount}** trial{'s' if trial_amount > 1 else ''} with {batch_range_str} "
            f"images each. A trial will end as soon as the model is "
            f"**{min_confidence}%** confident in its prediction."
        )
        update_box.info(update_str)


def process_inputs(
    trial_amount, min_attempts, max_attempts, min_confidence, warning_box
):
    if max_attempts < min_attempts:
        warning_box.error(
            f"The value for 'Max attempts per trial' must be equal to or higher "
            f"than the value for 'Minimal image threshold'. Adjusting."
        )
        max_attempts = min_attempts
    else:
        warning_box.text("")

    batch_range_str = (
        f"**{min_attempts}**"
        if min_attempts == max_attempts
        else f"**{min_attempts}** - **{max_attempts}**"
    )
    update_str = (
        f"The model will conduct **{trial_amount}** trial{'s' if trial_amount > 1 else ''} with {batch_range_str} "
        f"images each. A trial will end as soon as the model is "
        f"**{min_confidence}%** confident in its prediction."
    )

    st.session_state["update_str"] = update_str

    return update_str, max_attempts
