# backend.py - Complete Student Personalized Learning Chatbot with Gemini 1.5 Flash

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os
#import wolframalpha
#client = wolframalpha.Client('GR43744YJG')
from googleapiclient.discovery import build
import json
import google.generativeai as genai

load_dotenv()

# -------------------
# 1. LLM Configuration - Google Gemini 1.5 Flash
# -------------------
# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyCFkknyIeN8YYf4mdsoXeWeVQNRgdyZBzQ"))

# Initialize Google Gemini 1.5 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyCFkknyIeN8YYf4mdsoXeWeVQNRgdyZBzQ"),
    temperature=0.7,
    max_output_tokens=2048,
    top_p=0.8,
    top_k=40
)

# -------------------
# 2. Educational Tools with API Keys
# -------------------

# DuckDuckGo Search (no API key needed)
search_tool = DuckDuckGoSearchRun(region="us-en")

# Wikipedia API (no API key needed)
#wikipedia = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=500)
#wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

@tool
def wolfram_alpha_query(query: str, wolfram_client=None) -> dict:
    """Query Wolfram Alpha for computational answers, math, science, and factual data."""
    try:
        res = wolfram_client.query(query)
        result = ""
        for pod in res.pods:
            for sub in pod.subpods:
                if hasattr(sub, 'plaintext') and sub.plaintext:
                    result += f"{pod.title}: {sub.plaintext}\n"
        return {"result": result} if result else {"error": "No results found"}
    except Exception as e:
        return {"error": str(e)}


# YouTube Data API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "AIzaSyBBI7qCzOg1mUTwxjk9acnOVcknBYjhDPY")
youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)


@tool
def get_educational_videos(topic: str, max_results: int = 5) -> dict:
    """Search for educational videos on YouTube for a given topic."""
    try:
        request = youtube_service.search().list(
            q=f"{topic} educational tutorial",
            part="snippet",
            maxResults=max_results,
            type="video",
            videoDuration="medium",
            relevanceLanguage="en",
            order="relevance"
        )
        response = request.execute()

        videos = []
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            channel = item['snippet']['channelTitle']
            videos.append({
                'title': title,
                'channel': channel,
                'url': f'https://www.youtube.com/watch?v={video_id}'
            })

        return {"videos": videos}
    except Exception as e:
        return {"error": str(e)}


# OpenStax API (free educational resources)
@tool
def get_openstax_content(subject: str, topic: str = "") -> dict:
    """Fetch educational content from OpenStax textbooks."""
    try:
        # This is a simplified example - OpenStax doesn't have a direct public API
        # You would need to use their content via web scraping or their content API if available
        subjects = {
            "math": ["Algebra", "Calculus", "Statistics"],
            "science": ["Physics", "Chemistry", "Biology"],
            "social studies": ["History", "Economics", "Psychology"],
            "english": ["Literature", "Grammar", "Writing"],
            "computer science": ["Programming", "Algorithms", "Data Structures"]
        }

        if subject.lower() in subjects:
            if topic:
                return {
                    "subject": subject,
                    "topic": topic,
                    "resources": [f"OpenStax {subject} Textbook - Chapter on {topic}"]
                }
            else:
                return {
                    "subject": subject,
                    "topics": subjects[subject.lower()],
                    "resources": [f"OpenStax {subject} Textbook"]
                }
        else:
            return {"error": f"Subject {subject} not found in available resources"}
    except Exception as e:
        return {"error": str(e)}


# Khan Academy API (example implementation)
@tool
def get_khan_academy_resources(topic: str) -> dict:
    """Get learning resources from Khan Academy for a specific topic."""
    try:
        # Khan Academy doesn't have a direct public API for content retrieval
        # This is a placeholder implementation
        return {
            "topic": topic,
            "resources": [
                f"Khan Academy video lessons on {topic}",
                f"Khan Academy practice exercises on {topic}",
                f"Khan Academy articles on {topic}"
            ]
        }
    except Exception as e:
        return {"error": str(e)}


# Calculator tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations: add, subtract, multiply, divide."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "operation": f"{first_num} {operation} {second_num}",
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}


# Dictionary API for word definitions
@tool
def get_word_definition(word: str) -> dict:
    """Get definition and examples for a word."""
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        if response.status_code == 200:
            data = response.json()
            return {
                "word": word,
                "definitions": data[0]['meanings'][0]['definitions'][:3]  # First 3 definitions
            }
        else:
            return {"error": f"Word '{word}' not found"}
    except Exception as e:
        return {"error": str(e)}


# Combine all tools
tools = [
    search_tool,
    #wiki_tool,
    wolfram_alpha_query,
    calculator,
    get_educational_videos,
    get_openstax_content,
    get_khan_academy_resources,
    get_word_definition
]

llm_with_tools = llm.bind_tools(tools)


# -------------------
# 3. State Definition
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph Construction
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 7. Helper Functions
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


# -------------------
# 8. Example Usage
# -------------------
if __name__ == "__main__":
    # Example conversation
    config = {"configurable": {"thread_id": "1"}}

    # Initial message
    messages = [HumanMessage(content="Can you help me learn about quadratic equations?")]
    result = chatbot.invoke({"messages": messages}, config)

    print("Bot response:", result["messages"][-1].content)

    # Follow-up question
    messages = [HumanMessage(content="Can you show me some video tutorials?")]
    result = chatbot.invoke({"messages": messages}, config)

    print("Bot response:", result["messages"][-1].content)