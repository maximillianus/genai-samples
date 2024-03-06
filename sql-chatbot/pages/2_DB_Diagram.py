import streamlit as st
from PIL import Image

st.set_page_config(page_title="Database Diagram", page_icon="📈")

st.markdown("# Database Diagram")
st.sidebar.header("Database Diagram")

st.write(
    """
    This page lets you see Chinook DB diagram. For more information of Chinook database, please visit [Chinook Github](https://github.com/lerocha/chinook-database?tab=readme-ov-file)!
"""
)

import streamlit as st


def main():
    st.title("Chinook Data Model ERD")

    # Load image
    image = Image.open("pages/chinook-er-diagram.jpg")

    # Display image
    st.image(image, caption="Chinook DB", use_column_width=True)

if __name__ == "__main__":
    main()
