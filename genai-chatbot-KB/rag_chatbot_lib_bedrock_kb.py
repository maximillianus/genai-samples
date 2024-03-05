import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

from botocore.client import Config
import boto3


REGION_NAME = "us-east-1"
BEDROCK_KB_ID = "8RG8DF3KZL"

def get_llm():

    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION_NAME
    )

    CLAUDE_INSTANT = "anthropic.claude-instant-v1"
    CLAUDE_V2 = "anthropic.claude-v2"
    CLAUDE_V2_1 = "anthropic.claude-v2:1"

    model_kwargs = { #AI21
        "max_tokens_to_sample": 2000,
        "temperature": 0,
        "top_k": 10
        }

    llm = Bedrock(
        client=bedrock_client,
        model_id=CLAUDE_INSTANT, #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude

    return llm

def get_retriever():

    bedrock_agent_runtime = boto3.client(
        service_name="bedrock-agent-runtime",
        region_name=REGION_NAME
    )

    retriever = AmazonKnowledgeBasesRetriever(
        client=bedrock_agent_runtime,
        knowledge_base_id=BEDROCK_KB_ID,
        retrieval_config={
            "vectorSearchConfiguration": {"numberOfResults": 2}},
    )

    return retriever

def get_memory(): #create memory for this chat session

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        ai_prefix="Assistant",
        human_prefix="Human",
        return_messages=True
    ) #Maintains a history of previous messages

    return memory


def build_prompt():
    template = '''


Human: You are Claudia, a virtual assistant created by Bina Nusantara (Binus) university.
You are having conversation with a human who is either a student, staff, or external audience who are interested in Binus University.
You are friendly, helpful and polite.
You answer only in Bahasa Indonesia.
Do not hallucinate. You may say don't know if you don't know the answer.
Be concise when answering.
You can help human to do language translation if asked to do so.

You only answer questions related to following topics
Here are topics that you have information with, enclosed in <topic> tag.
<topic>
Bina Nusantara
Binus
internship
university
students
kalendar kuliah
jadwal ujian
</topic>

Answer topics-related question only based on provided context below
<context>
{context}
</context>

This is the chat history. Consider this past conversation also when answering.
<chat_history>
{chat_history}
</chat_history>

You also know other topics beyond what are listed in <topic> tag but you must be concise and limited when answering these out-of-scope topics.

You must not include any XML tags in your answer or response.

Here is human's next input, enclsoed in <input> tag.
<question>
{question}
</question>

Assistant:
'''
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            'chat_history',
            'context',
            'question',
        ]
    )

    return prompt

def get_rag_chat_response(input_text, memory, retriever): #chat client function

    llm = get_llm()

    prompt = build_prompt()

    chain_type_kwargs = {
        "verbose": True,
        "prompt": prompt,
        "memory": memory
    }

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs=chain_type_kwargs
    )

    response = qa({"query": input_text})

    result = response["result"]
    source_refs = []
    for source in response["source_documents"]:
        source_refs.append(source.metadata["location"]["s3Location"]["uri"])
    source_contents = []
    for source in response["source_documents"]:
        source_contents.append(source.page_content[:250])

    return result, source_refs, source_contents


