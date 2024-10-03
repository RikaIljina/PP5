import os
import base64
from datetime import datetime
import joblib
import streamlit as st
import tempfile


def download_dataframe_as_csv(dfs, caps):

    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".csv"
    ) as temp:
        for df, cap in zip(dfs, caps):
            temp.write(cap)
            temp.write("\r\n")
            df.to_csv(temp, mode="a", index=True)

        temp.flush()
        temp_name = temp.name

    with open(temp_name, "rb") as df_csv_temp:
        csv_content = b"sep=,\r\r\n"
        csv_content += df_csv_temp.read()

        b64 = base64.b64encode(csv_content).decode()

    button_css = '<button class="download-btn-2">'

    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    href = f"""{button_css}<a style="text-decoration: none; color: black;" href="data:file/csv;base64,{b64}" download="Report_{datetime_now}.csv" target="_blank">Download Report</a></button>"""
    os.remove(temp_name)

    return href  # csv_content, f"Report_{datetime_now}.csv"


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)
