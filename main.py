from finalize import graph

#input
max_customers = 3
topic = "Write a comment about the latest iphone 17"
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
final_report= final_state.values.get("final_report")
print(final_report)