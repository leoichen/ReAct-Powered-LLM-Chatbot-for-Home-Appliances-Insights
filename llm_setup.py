# import env vars and libraries
from dotenv import load_dotenv
load_dotenv()

from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.tools import tool
from langchain_core.tools import Tool

from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain_community.utilities import GoogleSearchAPIWrapper

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain import hub

from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from transformers import GPT2TokenizerFast

from operator import itemgetter
from langchain.agents.format_scratchpad import format_to_openai_function_messages
import tiktoken

from langchain_core.prompt_values import ChatPromptValue
from langchain.agents.output_parsers.openai_functions import OpenAIFunctionsAgentOutputParser
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate

from langchain_core.messages import AIMessage, HumanMessage


# set up local vector DB
loader = CSVLoader(file_path="llm/data/products_1.csv")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="]",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

splits = text_splitter.split_documents(docs)
retriever = FAISS.from_documents(splits, OpenAIEmbeddings()).as_retriever()


# prompt setup
template = """You are an support chatbot for the employees of a e-commerce website company called PartSelect
that sells parts for appliances. Your goal is to answer the employees' questions on the appliances that 
the company sells. 

Even though the company sells a lot of different appliances, your focus will be on all dishwashers and all
refrigerators. If there are questions regarding non-dishwashers or non-refrigerators, answer with
"I'm currently not equipped to deal with that kind of appliance. Please refer to the PartSelect website
for more information".

Before you answer any questions, make sure you search the database for information regarding that
specific product. If the database returns nothing, search the PartSelect website for information. 
If that returns nothing, ask the user to double check the provided model number.

If the user asks for a list of things, you don't need to return it all. Make sure you are able to
complete your sentences.

You must check the token size of the message and truncuate it if
it is too large. You must do this before you generate an response!

Answer:
"""
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template,
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

try:
    chat_history
except NameError:
    chat_history = []

# tools setup
products_df = pd.read_csv('llm/data/products_1.csv')
product_urls_df = pd.read_csv('llm/data/products_urls.csv')

retriever_tool = create_retriever_tool(
    retriever,
    "search",
    "Search the database for information about appliances. For any questions about the appliances, you must start with this tool!")

@tool
def search_product_database(model_name_or_number:str) -> str:
    """Use this to search through the database given the model name or number."""
    
    filtered_df = products_df[products_df['Model Number'] == model_name_or_number]    

    if len(filtered_df) == 0:
        filtered_df = products_df[products_df['Model Name'] == model_name_or_number]
        
    return filtered_df.to_string(header = False, index = False) if len(filtered_df) > 0 else ''

@tool
def search_for_url_with_model_number(model_number: str) -> str:
    """Use this to search for the URL for the specific product based on the model number."""

    filtered_df = product_urls_df[product_urls_df['Model Number'] == model_number]
    
    return '\n'.join(filtered_df['Model URL'].values) if len(filtered_df) > 0 else ''

@tool
def search_for_url_with_model_name(model_name: str) -> list:
    """Use this to search for the URL for the specific product based on the model name."""

    filtered_df = product_urls_df[product_urls_df['Model Name'].str.lower() == model_name.lower()]
        
    return filtered_df['Model URL'].values if len(filtered_df) > 0 else []


@tool
def scrape_partselect_website(href):
    """If the database doesn't return anything, use this to search the PartSelect website for information."""
    try:
        root_url = 'https://www.partselect.com'
        curr_url = root_url + href
        curr_product_webpage = requests.get(curr_url)
        curr_product_sp = BeautifulSoup(curr_product_webpage.content, 'html.parser')

        # get title
        title = curr_product_sp.find('div', {'id': 'main'}).find('h1').text
        title = title.split('-')[0].strip()

        final_title_str = 'Title: ' + title + '\n'

        # get parts
        parts_arr = []
        page_start = 1
        while page_start < 5:
        #     print(page_start)
            curr_product_parts_webpage = requests.get(f'{curr_url}/parts?start={page_start}')
            curr_product_parts_sp = BeautifulSoup(curr_product_parts_webpage.content, 'html.parser')

            parts_elements = curr_product_parts_sp.select('div.mega-m__part > a')
            if len(parts_elements) == 0:
                break

            for part in parts_elements:
                cur_part_full_title = part['title'].split(" – Part Number: ")
                cur_part_title = cur_part_full_title[0]
                cur_part_manuf_number = cur_part_full_title[1]
                cur_part_part_number = part['href'][1:].split('-')[0]
                part_full_str = f"{{Part Title: {cur_part_title}, Manufacturing Number: {cur_part_manuf_number}, Part Number: {cur_part_part_number}}}"
                parts_arr.append(part_full_str)

            page_start += 1

        final_parts_str = 'Compatible Parts: {' + ' '.join(parts_arr) + '}\n'

        # get qas
        qas = curr_product_sp.find('div', {'id': 'QuestionsAndAnswersContent'})# > div.js-dataContainer > div > div.qna__question")

        qa_tags = qas.find_all('div', class_='qna__question')
        qa_arr = []
        for qa in qa_tags:
            qa_question = qa.find('div', class_='js-searchKeys').text
            qa_answer = qa.find('div', class_='qna__ps-answer__msg').text
            qa_str = f"{{Question: {qa_question}, Answer: {qa_answer}}}"
            qa_arr.append(qa_str)

        final_qas_str = 'Questions and Answers: {' + ' '.join(qa_arr) + '}\n'

        # get symptoms
        symptom_tags = curr_product_sp.find_all('a', class_='symptoms')
        symptoms_arr = []
        for tag in symptom_tags:
            symptoms_arr.append(tag.find('div', class_='symptoms__descr').text)

        final_symptoms_str = 'Symptoms: {' + ''.join(symptoms_arr) + '}\n'

        # all installations
        page_start = 1
        installation_arr = []
        while page_start < 5:

            webpage = requests.get(root_url + href+f'/instructions?start={page_start}')
            curr_product_install_sp = BeautifulSoup(webpage.content, 'html.parser')

            repair_stories = curr_product_install_sp.find_all('div', class_='repair-story')

            if len(repair_stories) == 0:
                break

            for story in repair_stories:
                installation_issue = story.find('div', class_='repair-story__title').text
                installation_how_to = story.find('div', class_='repair-story__instruction__content').text
                installation_str = f"{{Installation Issue: {installation_issue}, How to Repair: {installation_how_to}}}"
                installation_arr.append(installation_str)
            page_start += 1

        final_installations_str = 'Installations: {' + ''.join(installation_arr) + '}\n'

        return final_title_str + final_parts_str + final_qas_str + final_installations_str
    except:
        return ''
    
@tool
def scrape_all_urls(urls: list) -> str:
    """When you are provided with a list of URLs, use this to scrape through all of them.
    This is usually the case when you are provided with a model name."""    
    results = ''
    
    # only scraping first 3 URLS to not exceed token limit
    for url in urls[:3]:
        results += scrape_partselect_website(url)
    return results


# tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')
# @tool
# def reduce_msg_size(message: str) -> str:
#     """Use this to ensure message size is within token size. You must do this before 
#     generating an output for the user!"""

#     # Tokenize the text and count the tokens
#     tokens = tokenizer.encode(message)
#     token_count = len(tokens)
#     print(f"Token count: {token_count}")

#     # Define your token limit
#     TOKEN_LIMIT = 7000

#     # Check if the token count exceeds the limit
#     if token_count > TOKEN_LIMIT:
#         # Truncate the tokens to the limit
#         truncated_tokens = tokens[:TOKEN_LIMIT]
#         # Convert tokens back to text
#         message = tokenizer.decode(truncated_tokens)

#     return message


encoder = tiktoken.encoding_for_model("gpt-4")
def condense_prompt(prompt: ChatPromptValue) -> ChatPromptValue:
    """This is used to ensure the context provided to the LLM is within its token limits """
    token_limit = 6500
    messages = prompt.to_messages()
    ai_function_messages = messages[2:]
    
    if len(ai_function_messages) > 0:
        context = ai_function_messages[1].content
        encoded_context = encoder.encode(context)
        num_tokens = len(encoded_context)
        
        if num_tokens >= token_limit:
            truncated_encoded_context = encoded_context[:token_limit]
            messages[2:][1].content = encoder.decode(truncated_encoded_context)
    print('MESSAGE', messages)
    return ChatPromptValue(messages=messages)


# set up tools, agent and agent executor
tools = [search_product_database, scrape_partselect_website, scrape_all_urls,
        search_for_url_with_model_number, search_for_url_with_model_name]

llm = ChatOpenAI(model="gpt-4", temperature=0)
# llm.max_tokens = 7000

agent = (
    {
        "input": itemgetter("input"),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | condense_prompt
    | llm.bind_functions(tools)
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print('LLM Setup Complete!')


