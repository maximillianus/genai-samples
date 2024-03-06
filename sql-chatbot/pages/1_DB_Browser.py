import streamlit as st
import time
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, MetaData, Table
import pandas as pd

st.set_page_config(page_title="Database Browser", page_icon="📈")

st.markdown("# Database Browser")
st.sidebar.header("Database Browser")

st.write(
    """
    This page lets you browse Chinook database. For more information of Chinook database, please visit [Chinook Github](https://github.com/lerocha/chinook-database?tab=readme-ov-file)!
"""
)

# Function to connect to the SQLite database and fetch table names
def get_table_names(database_url):
    engine = create_engine(database_url)
    meta = MetaData()
    # meta.bind = engine
    meta.reflect(bind=engine)
    return meta.tables.keys()

# Function to fetch data from a table
def fetch_table_data(database_url, table_name):
    engine = create_engine(database_url)
    connection = engine.connect()
    table = Table(table_name, MetaData(), autoload_with=engine)
    result = connection.execute(table.select())
    colnames = result.keys()
    data = result.fetchall()
    connection.close()
    return colnames, data

# Main function to create Streamlit app
def main():

    # Database connection configuration
    database_url = "sqlite:///Chinook.db"
    table_names = get_table_names(database_url)


    # Sidebar widget to select table
    selected_table = st.selectbox("Select Table", table_names)

    # Fetch data from selected table
    table_columns, table_data = fetch_table_data(database_url, selected_table)

    # Display table data
    st.write(f"Displaying data from table: {selected_table}")
    df = pd.DataFrame(table_data, columns=[col for col in table_columns])
    st.dataframe(df)

if __name__ == "__main__":
    main()


