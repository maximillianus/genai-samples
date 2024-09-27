# GenAI Chatbot with Knowledge Base

This is GenAI Chatbot using Amazon Bedrock's Knowledge Base. You can store documents and files in the KNowledge Base to be queried using this app. The app's front-end is built with `streamlit`, back-end is built using `langchain`. Note that you have to update the Bedrock's Knowledge Base ID with your own ID to use this app properly. This app is using AWS region `us-east-` with Claude 3 Sonnet as the LLM.

## Running the app
1. Install the necessary library using:
   ```bash
   pip3 install -r requirements.txt
   ```
2. Run using streamlit command:
   ```bash
   streamlit run rag_chatbot_app.py
   ```
