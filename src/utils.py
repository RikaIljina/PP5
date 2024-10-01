
import streamlit as st


def update_info_box(trial_amount, min_attempts, max_attempts, min_confidence, update_box):
    if st.session_state['tr_input']:
        batch_range_str = (f"**{min_attempts}**" if min_attempts == max_attempts 
                               else f"**{min_attempts}** - **{max_attempts}**")
        update_str = (
                f"The model will conduct **{trial_amount}** trial{'s' if trial_amount > 1 else ''} with {batch_range_str} "
                f"images each. A trial will end as soon as the model is "
                f"**{min_confidence}%** confident in its prediction."
                )
        update_box.info(update_str)



def process_inputs(trial_amount, min_attempts, max_attempts, min_confidence, warning_box):
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
    
    return update_str, max_attempts
