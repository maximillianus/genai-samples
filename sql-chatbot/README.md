# SQL Chatbot Guide

Here is how to build SQL Chatbot App using Amazon Bedrock:
1. Build the database first using Chinook DB sample DB. Chinook DB explanation [here](https://github.com/lerocha/chinook-database). You can copy `Chinook_SQlite.sql` and run `build_sqlite3_db.sh`. Ensure you have sqlite3 in your machine (*usually preinstalled in Linux/Mac machine*).
2. Run app `streamlit run sql_chatbot_app.py` to run in default server port `8051`

Notes on SQL LLM:
- LangChain will generate SQL query based on database schema it reads.
- LangChain connects to a database, read it schema, and use it as context in its input token. The more complex a database schema is, the more number input tokens it will generate.
- Generated SQL query will then be run on the database itself, so number of data size and DB's server spec will affect how fast a query will run.