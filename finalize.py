from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.types import Send
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o", temperature=0) 

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

class InterviewState(MessagesState):
    max_num_turns:int 
    context: Annotated[List, operator.add]
    customer: Customer
    interview:str
    sections:list

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
    print("search_web")

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
    print("search_wikipedia")
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
    print("generate_answer")
    customer = state["customer"]
    messages = state["messages"]
    context = state["context"]

    system_message = answer_instructions.format(goals= customer.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)

    answer.name = "customer"

    return{"messages":[answer]}


from langchain_core.messages import get_buffer_string

def save_interview(state: InterviewState):
    print("save_interview")
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview":interview}


def route_messages (state:InterviewState, name:str="customer"):
    print("route_messages")
    messages = state["messages"]
    max_num_turns= state.get('max_num_turns',2)

    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
    print(num_responses)
    if num_responses >= max_num_turns:
        return 'save_interview'
    
    last_question = messages[-2]

    if "Thank you so much for your help!" in last_question.content:
        return "save_interview"
    return "ask_question"

#write review
section_writer_instructions = """ 
You are an expert technical writer.
Your task is to create a short, easily digestible section of a report based on a set of source documents.
1. Analyze the content of the source documents:
- The name of each source document is at the start of the document, with the <Document tag.
2. Create a report structure using markdoen formatting:
- Use ## for the section title
- Use ### for sub-section headers
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)
4. Make your title engaging based upon the focus area of the analyst: 
{focus}
5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed
"""

def write_section(state:InterviewState):
    print("write_section")
    interview = state["interview"]
    context=state["context"]
    customer=state["customer"]

    system_message = section_writer_instructions.format(focus=customer.description)
    section = llm.invoke([SystemMessage(content=system_message)]+ [HumanMessage(content=f"Use this source to write your section: {interview}")])

    return {"sections":[section.content]}

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section",write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', "save_interview"])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

class ResearchGraphState(TypedDict):
    topic:str
    max_customers:int
    human_analyst_feedback:str
    customers:List[Customer]
    sections: Annotated[list, operator.add]


def inititate_all_interviews(state:ResearchGraphState):
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_customers"
    else:
        topic=state["topic"]
        return [Send("conduct_interview",{"customer":customer,"messages":[HumanMessage(content=f"So you said you were writing an article on {topic}?")]})for customer in state["customers"]]
    

builder = StateGraph(ResearchGraphState)
builder.add_node("create_customers", create_customers)
builder.add_node("human_feedback",human_feedback)
builder.add_node("conduct_interview",interview_builder.compile())

builder.add_edge(START, "create_customers")
builder.add_edge("create_customers", "human_feedback")
builder.add_conditional_edges("human_feedback", inititate_all_interviews,["create_customers","conduct_interview"])
builder.add_edge("conduct_interview",END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

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
# print(state)
while True:
    user_approval = input("Do you want to revise the customers? (yes/no)")

    if user_approval.lower() == "yes":
        feedback = input("provide feedback: ")
        graph.update_state(thread, {"human_analyst_feedback":feedback}, as_node="human_feedback")
        # state = graph.get_state(thread)
        # print(state)
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

for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)
    
final_state = graph.get_state(thread)
sections= final_state.values.get("sections")
for s in sections:
    print(s)