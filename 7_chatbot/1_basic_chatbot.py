# Tutorial: https://www.youtube.com/watch?v=KU_FDwwL5_s&list=PLNIQLFWpQMRXmns-7UarmPIR6DN7bgEzZ&index=27
# This is a single-turn chatbot. 
# It has not memory. 
# Every time you send a new message, it starts the graph over again. 

from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1")

# BasicChatState is a TypedDict that defines the state of the chatbot
# add_message is a method provided by langgraph to add messages to the state
class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

# Define the chatbot function
# the llm.invoke() passes the state ["messages"] to the llm
def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

#------------------------------------------------------------------------------
# CREATE THE GRAPH - TYPICALLY GOES IN SEPARATE ORCHESTRATOR FILE
#------------------------------------------------------------------------------

# StateGraph is a class provided by langgraph to define the graph of the chatbot
graph = StateGraph(BasicChatState)

# Add the chatbot node to the graph
graph.add_node("chatbot", chatbot)

# Set the entry point of the graph to the chatbot node
graph.set_entry_point("chatbot")

# Add an edge (pathway) from the chatbot node to the end of the graph
# This edge (pathway) creates a pathway from chatbot agent/node to END (of chat)
graph.add_edge("chatbot", END)

# Compile the graph into an application
app = graph.compile()

#------------------------------------------------------------------------------
# CREATE THE MAIN CHAT LOOP - TYPICALLY GOES IN SEPARATE MAIN.py FILE
#------------------------------------------------------------------------------

# This allows user to chat with the chatbot in the terminal. 
    # user_input collectes the user's inout.
    # input is a method provided by langgraph to collect the user's input.
    # "User: " is what will display in the terminal.
    # if the user inputs "exit" or "end", the loop will break.
    # otherwise, the user's input will be added to the state and the graph will be invoked.
    # the result will be printed to the terminal.

while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        print(result)
