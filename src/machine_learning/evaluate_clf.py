import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation():
    return (
        load_pkl_file(f'outputs/model_final_eval.pkl'),
        load_pkl_file(f'outputs/hyperparam_values.pkl'),
        )

def load_reports():
    return {'test': load_pkl_file(f'outputs/test_class_report.pkl'),
            'live': load_pkl_file(f'outputs/live_class_report.pkl'),
            'live_batches': load_pkl_file(f'outputs/live_class_report_batches.pkl'),       
    }