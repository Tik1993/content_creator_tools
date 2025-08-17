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
    interview:str
    sections:list

question_instructions="""You are an analyst tasked with interviewing an expert to learn about a specific topic.
Your goal is boil down to interesting and specific insights related to your topic.
1. Interesting: Insights that people will find surprising or non-obvious
2. Specific: Insights the avoid generalities and include specific examples from the expert.
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

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", END)

memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory)

#input
topic = " the latest snowbaord binding Fuse"
thread = {"configurable":{"thread_id":"2"}}
customer = Customer(name="Jake Thompson", occupation="Professional Snowboarder", age=28, description="Jake is a professional snowboarder who competes in various international events. He is always on the lookout for the latest gear that can give him an edge in competitions. His main focus is on performance and durability, as he needs equipment that can withstand rigorous use and enhance his skills on the slopes.")
messages= [HumanMessage(f"So you said you were writing an article on {topic}?")]
max_num_turns=2
interview = interview_graph.invoke({"customer":customer, "messages":messages, "max_num_turns":max_num_turns}, thread)
print(interview)

