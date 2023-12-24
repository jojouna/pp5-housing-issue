import streamlit as st


class MultiPage:
    """
    Class to generate multiple Streamlit pages
    using an object oriented approach
    """

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon=":page_facing_up:"
        )

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        """
        Sets title of application, creates side bar
        with radio button for page selection, and runs
        the function associated with
        """
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages,
                                format_func=lambda page: page['title'])
        page['function']()
