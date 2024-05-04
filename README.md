# ReAct Powered LLM Chatbot for Home Appliances Insights
This repo contains the code for the development of an agentic GPT-4 chatbot with a focus on the 180,000+ dishwashers/refrigerators on PartSelect.com. The motivation is to build a chatbot for internal use for the employees of PartSelect.com. It can provide insights on the 180,000+ dishwashers/refrigerators, allowing the employees to quickly acquire information regarding company products. This enables them to more efficiently and effectively assist customers.

## Application Infrastructure
The infrastucture of the app can be seen in the image below. The thought process is to create a agentic LLM that is equipped with custom tools. The custom tools allows the LLM to not only search through relational databases but also scrape the PartSelect.com website for real-time information. The web scraper tool provides the LLM agent with the ability to attain the most up-to-date information and also minimizes storage costs as I don't need to store information on all 180,000 appliances.

The product database contains information on the appliancs of the 50 most searched appliances, helping to minimize latency that occurs with web scraping.

## Code Files
There are two code files for this project:

_llm_main.py_

_llm_setup.py_
