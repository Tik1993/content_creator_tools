from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0) 

from typing import List
from typing_extensions import TypedDict
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

class Perspective(BaseModel):
    customers: List[Customer] = Field(
        description="Comprehensive list of customers with their occupation and age"
    )

class GenerateCustomersState(TypedDict):
    topic: str #Research product
    max_customers: int # Number of customer
    human_analyst_feedback: str # Human feedback
    customers: List[Customer] # Customers writing review

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

customer_instructions = """You are tasked with creating a set of AI customer personas. Follow these instructions carefully
1. First, review the product topic: {topic}

2. Examine any editoral feedback that has been optionally provided to guide creation of the analysts: {human_analyst_feedback}

3. Determin the most interesting themes based upon documents and/or feedback above.

4. Pick the top {max_customers} themes.

5. Assign one analyst to each theme."""

def create_customers(state: GenerateCustomersState):
    topic = state["topic"]
    max_customers= state["max_customers"]
    human_analyst_feedback = state.get("human_analyst_feedback","")

    structured_llm = llm.with_structured_output(Perspective)

    system_message = customer_instructions.format(topic=topic,human_analyst_feedback=human_analyst_feedback, max_customers=max_customers)

    customers = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of customers")])

    return {"customers":customers.customers}

def human_feedback(state:GenerateCustomersState):
    pass

def should_continue(state:GenerateCustomersState):
    print("should_continue")
    human_analyst_feedback= state.get('human_analyst_feedback',None)
    if human_analyst_feedback:
        return "create_customers"
    return END

builder = StateGraph(GenerateCustomersState)
builder.add_node("create_customers",create_customers)
builder.add_node("human_feedback", human_feedback)
builder.add_edge(START, "create_customers")
builder.add_edge("create_customers","human_feedback")
builder.add_conditional_edges("human_feedback",should_continue, ["create_customers",END])

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer= memory)

#input
max_customers = 3
topic = " the latest snowbaord binding Fuse"
thread = {"configurable":{"thread_id":"1"}}

for event in graph.stream({"topic":topic, "max_customers":max_customers}, thread, stream_mode="values"):
    customers = event.get('customers','')
    if customers:
        for c in customers:
            print(f"Name: {c.name}")
            print(f"Occupation: {c.occupation}")
            print(f"Age: {c.age}")
            print(f"Description: {c.description}")
            print("-"*50)

state = graph.get_state(thread)
print(state)
while True:
    user_approval = input("Do you want to revise the customers? (yes/no)")

    if user_approval.lower() == "yes":
        feedback = input("provide feedback: ")
        graph.update_state(thread, {"human_analyst_feedback":feedback}, as_node="human_feedback")
        state = graph.get_state(thread)
        print(state)
        for event in graph.stream(None, thread, stream_mode="values"):
            customers = event.get('customers','')
            if customers:
                for c in customers:
                    print(f"Name: {c.name}")
                    print(f"Occupation: {c.occupation}")
                    print(f"Age: {c.age}")
                    print(f"Description: {c.description}")
                    print("-"*50)
    if user_approval.lower() == 'no':
        graph.update_state(thread, {"human_analyst_feedback":None}, as_node="human_feedback")
        break

final_state = graph.get_state(thread)
print(final_state.values.get("customers"))
