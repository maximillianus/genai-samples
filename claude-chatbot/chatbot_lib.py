import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

REGION_NAME = "us-east-1"

def get_llm():

    CLAUDE_INSTANT = "anthropic.claude-instant-v1"
    CLAUDE_V2 = "anthropic.claude-v2"
    CLAUDE_V2_1 = "anthropic.claude-v2:1"

    model_kwargs = { # Titan Lite
        "max_tokens_to_sample": 384,
        "temperature": 0, # alter only temperature or top_p
        "top_k": 5,
        "stop_sequences": ["User:"],
    }

    llm = Bedrock(
        region_name=REGION_NAME, #sets the region name (if not the default)
        model_id=CLAUDE_INSTANT,
        model_kwargs=model_kwargs) #configure the properties for Amazon Titan

    return llm


def get_memory(): #create memory for this chat session

    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    #this allows us to maintain the "big picture" of a long-running conversation
    llm = get_llm()

    # Use Conversation BufferWindowMemory to only limit to previous 2 conversations

    memory = ConversationBufferWindowMemory(
        llm=llm,
        return_messages=True,
        k=4,
        ai_prefix="Assistant",
        human_prefix="Human") #Maintains a summary of previous messages

    return memory

def build_prompt_template():
    template = """


Human: This is a friendly conversation between a human and a virtual assistant named Claudia.
Claudia is friendly, polite, and helpful.
Be concise when answering and politely say do not know if you don't have the answer.
Do not be robotic when answering, be humane. Respond only in Bahasa Indonesia. Never include any xml tags when responding.

Current conversation
<history>
{history}
</history>

Human's next reply below
<human_reply>{input}</human_reply>
Assistant:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

    return PROMPT

def build_prompt_template_for_binus():
    template = """

System: This is a friendly conversation between a user and a virtual assistant named Titan.
Titan is a virtual assistant created by Binus (Bina Nusantara) university.
Titan may do small talk with the user.
Be concise when answering and politely say do not know if you don't have the answer.
When answering specific facts about Binus, always remind the user to check latest information in Binus website.
Do not be robotic when answering, be humane. Respond only in Bahasa Indonesia.

Current conversation
{history}

User: {input}
Bot:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

    return PROMPT

def get_chat_response(input_text, memory): #chat client function

    llm = get_llm()

    PROMPT = build_prompt_template()

    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, #using the Bedrock LLM
        memory = memory, #with the summarization memory
        prompt=PROMPT,
        verbose = True #print out some of the internal states of the chain while running
    )

    chat_response = conversation_with_summary.predict(input=input_text) #pass the user message and summary to the model

    return chat_response
