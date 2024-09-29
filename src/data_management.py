import numpy as np
import pandas as pd
import os
import streamlit.components.v1 as components
import base64
from datetime import datetime
import joblib
import streamlit as st
import tempfile

def download_dataframe_as_csv(dfs, caps):
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv') as temp:
        for df, cap in zip(dfs, caps):
            temp.write(cap)
            temp.write('\r\n')
            df.to_csv(temp, mode='a', index=True)

            

        temp.flush()
        temp_name = temp.name
        
    with open(temp_name, 'rb') as df_csv_temp:
        csv_content = b'sep=,\r\r\n'
        csv_content += df_csv_temp.read() 
        
        b64 = base64.b64encode(csv_content).decode()

    
    button_css = '<button class="download-btn-2">'
    
    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    href = f'''{button_css}<a style="text-decoration: none; color: black;" href="data:file/csv;base64,{b64}" download="Report_{datetime_now}.csv" target="_blank">Download Report</a></button>'''
    os.remove(temp_name)
    
    
    #download_button(b64, f"Report_{datetime_now}.csv") #href, f"data:file/csv;base64,{b64}"
    return href # csv_content, f"Report_{datetime_now}.csv"


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)


# Adapted from https://discuss.streamlit.io/t/automatic-download-select-and-download-file-with-single-button-click/15141/4
def download_button(b64, download_filename):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    """
    # if isinstance(object_to_download, pd.DataFrame):
    #     object_to_download = object_to_download.to_csv(index=False)

    # # Try JSON encode for everything else
    # else:
    #     object_to_download = json.dumps(object_to_download)

    # try:
    #     # some strings <-> bytes conversions necessary here
    #     b64 = base64.b64encode(object_to_download.encode()).decode()

    # except AttributeError as e:
    #     b64 = base64.b64encode(object_to_download).decode()

    dl_link = f"""
    <html>
    <head>
    <title>Start Auto Download file</title>
    <script src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
    <script>
    $('<a href="data:text/csv;base64,{b64}" download="{download_filename}" target="_blank">')[0].click()
    </script>
    </head>
    </html>
    """
    st.write(dl_link)
    components.html(dl_link, width=0, height=0)
    
    return dl_link