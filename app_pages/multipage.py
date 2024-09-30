import streamlit as st



# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name
        with open('./files/wave.css') as f:
            self.css = f.read()

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🐺",
            )  # You may add an icon, to personalize your App
        # check links below for additional icons reference
        # https://docs.streamlit.io/en/stable/api.html#streamlit.set_page_config
        # https://twemoji.maxcdn.com/2/test/preview.html

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages, format_func=lambda page: page['title'])  #.radio('Menu', self.pages, format_func=lambda page: page['title'])

        st.write(f"<style>{self.css}</style>", unsafe_allow_html=True)
        
        page['function']()
