import streamlit as st


# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🖥️")  # You may add an icon, to personalize your App
        # check links below for additional icons reference
        # https://docs.streamlit.io/en/stable/api.html#streamlit.set_page_config
        # https://twemoji.maxcdn.com/2/test/preview.html

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages, format_func=lambda page: page['title'])  #.radio('Menu', self.pages, format_func=lambda page: page['title'])
        #st.sidebar.page_link("pages/page_dataset_assessment.py", label="Page DA", icon="1️⃣")
        css = """
        <style>
        [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p {
            /* Styles for the label text for checkbox and toggle */
            font-size: 1.2rem;
            font-weight: bold;
            width: 100vw;
            margin-left: 1rem;
            
        }

        [data-baseweb="checkbox"] div {
            /* Styles for the slider container */
            height: 1.6rem;
            width: 2rem;
        }
        [data-baseweb="checkbox"] div div {
            /* Styles for the slider circle */
            height: 1.8rem;
            width: 1.8rem;
        }
        [data-testid="stCheckbox"] label span {
            /* Styles the checkbox */
            height: 1.2rem;
            width: 1.2rem;
            margin-top: 0.5rem;
        }
        [data-testid="stRadio"] p {
            height: 2rem;
            font-size: 1.2rem;
            font-weight: bold;
        }
        
        </style>
        """
        st.write(css, unsafe_allow_html=True)
        
        page['function']()
