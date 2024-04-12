from llm.llm_setup import agent_executor, chat_history
from langchain_core.messages import AIMessage, HumanMessage

def llm_query(user_query):

    result = agent_executor.invoke({
        "input": user_query,
        "chat_history": chat_history
    })
    
    chat_history.extend(
        [
            HumanMessage(content = user_query),
            AIMessage(content = result['output'])
        ]
    )
#     result = agent_with_chat_history.invoke(
#     {"input": user_query},
#     config={"configurable": {"session_id": "<foo>"}},
# )
    
    return result['output']
    
