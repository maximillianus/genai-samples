from operator import itemgetter
import streamlit as st #all streamlit commands will be available through the "st" alias
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.llms.bedrock import Bedrock
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

DATABASE_URI = "sqlite:///Chinook.db"


def get_llm():

    REGION_NAME = "us-east-1"

    TITAN_LITE = "amazon.titan-text-lite-v1"
    TITAN_EXPRESS = "amazon.titan-text-express-v1"

    model_kwargs = { # Titan Lite
        "maxTokenCount": 256,
        "temperature": 0,
        "topP": 0.25,
        "stopSequences": ["User:"],
    }

    llm = Bedrock(
        region_name=REGION_NAME, #sets the region name (if not the default)
        model_id=TITAN_EXPRESS,
        model_kwargs=model_kwargs) #configure the properties for Amazon Titan

    return llm

def get_chat_response(input_text): #chat client function

    db = SQLDatabase.from_uri(DATABASE_URI)

    llm = get_llm()

    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)

    answer_prompt = PromptTemplate.from_template(
    """
    System: Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    User: {question}
    SQL Query: {query}
    SQL Result: {result}
    Bot: """
    )

    answer = answer_prompt | llm | StrOutputParser()
    chain = (
        RunnablePassthrough.assign(query=write_query).assign(
            result=itemgetter("query") | execute_query
        )
        | answer
    )

    response = chain.invoke({"question": input_text})

    return response


st.set_page_config(page_title="SQL Chatbot Demo") #HTML title
st.title("SQL Chatbot Demo") #page
st.sidebar.header("SQL Chatbot")
st.markdown("""
Chatbot using Amazon Titan to answer Business Intelligence queries.\n
The example DB used here is SQlite Chinook DB. You can browse the database and the datamodel using links in left sidebar.
\nExample queries you can try:
- Give me top 3 countries with most purchase along with number of purchase
- Which artist has the most number of albums?
- Give me the most popular song genre
""")


# if 'memory' not in st.session_state: #see if the memory hasn't been created yet
#     st.session_state.memory = get_memory() #initialize the memory


if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [] #initialize the chat history


#Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history: #loop through the chat history
    with st.chat_message(message["role"]): #renders a chat line for the given role, containing everything in the with block
        st.markdown(message["text"]) #display the chat content



input_text = st.chat_input("Ask questions about your data here. Example: Give me top sales record.") #display a chat input box

if input_text: #run the code in this if block after the user submits a chat message

    with st.chat_message("user"): #display a user chat message
        st.markdown(input_text) #renders the user's latest message

    st.session_state.chat_history.append({"role":"user", "text":input_text}) #append the user's latest message to the chat history

    chat_response = get_chat_response(input_text=input_text)

    with st.chat_message("assistant"): #display a bot chat message
        st.markdown(chat_response) #display bot's latest response

    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) #append the bot's latest message to the chat history

