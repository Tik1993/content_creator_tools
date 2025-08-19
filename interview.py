from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0) 

import operator
from typing import List, Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class Customer(BaseModel):
    name:str=Field(
        description="Name of the customer"
    )

    occupation: str = Field(
        description="Occupration of the customer"
    )

    age: int = Field(
        description="customer's age"
    )

    description: str = Field(
        description="Description of the customer focus, concerns, and motives.",
    )
    @property
    def persona(self)-> str:
        return f"Name: {self.name}\nOccupation:{self.occupation}\nAge:{self.age}\nDescription: {self.description}\n"

class InterviewState(MessagesState):
    max_num_turns:int 
    context: Annotated[List, operator.add]
    customer: Customer
    # interview:str
    # sections:list

class SearchQuery(BaseModel):
    search_query:str = Field(None, description="Search query for retrieval.")

question_instructions="""You are an analyst tasked with interviewing an customer to learn about a specific topic.
Your goal is boil down to interesting and specific insights related to your topic.
1. Interesting: Insights that people will find surprising or non-obvious
2. Specific: Insights the avoid generalities and include specific examples from the customer.
Here is your topic of focus and set of goals: {goals}
Begin by introding yourself using a name that fits your persona, and then ask your question.
Continue to ask questions to drill down and refine your understanding of the topic.
When you are satisfied with your understanding, complete the interview with "Thank you so much for your help!
Remember to stay in character throughout your response, reflecting the persona and goals prvided to you.
"""

def generate_question(state:InterviewState):
    customer=state["customer"]
    messages = state["messages"]

    system_message = question_instructions.format(goals=customer.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)

    return {"messages":[question]}


#Searching information for answer
# Web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_search = TavilySearchResults(max_results=3)

# Wikipedia search tool
from langchain_community.document_loaders import WikipediaLoader

#Search query writing
search_instructions = SystemMessage(content=f""" You will be given a conversation between an analyst and a customer.
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
First, analyze the full converstion.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well-structured web search query""")

def search_web(state:InterviewState):

    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])

    search_docs = tavily_search.invoke(search_query.search_query)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context":[formatted_search_docs]}

def search_wikipedia(state:InterviewState):
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])

    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context":[formatted_search_docs]}

#Generate answer
answer_instructions = """You are a customer being interviewed by an analyst.
Here is analyst area of focus:{goals}.
Your goal is to answer a question posed by the interviewer.
To answer question, use this context: {context}
When answering questions, follow these guidelines:
1. Use only the information provided in the context.
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.
3. The context contains sources at the topic of each individual document.
4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1].
5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list:[1] assistant/docs/llama3_1.pdf, page 7 
And skip the addition of the brackets as well as the Document source preamble in your citation.
"""

def generate_answer(state:InterviewState):
    customer = state["customer"]
    messages = state["messages"]
    context = state["context"]

    system_message = answer_instructions.format(goals= customer.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)

    answer.name = "customer"

    return{"messages":[answer]}

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_edge("answer_question", END)

memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory)

#input
topic = "which binding from Union would you perfer"
thread = {"configurable":{"thread_id":"2"}}
customer = Customer(name="Jake Thompson", occupation="Professional Snowboarder", age=28, description="Jake is a professional snowboarder who competes in various international events. He is always on the lookout for the latest gear that can give him an edge in competitions. His main focus is on performance and durability, as he needs equipment that can withstand rigorous use and enhance his skills on the slopes.")
messages= [HumanMessage(f"So you said you were writing an article on {topic}?")]
max_num_turns=2
interview = interview_graph.invoke({"customer":customer, "messages":messages, "max_num_turns":max_num_turns}, thread)

for m in interview["messages"]:
    m.pretty_print()

