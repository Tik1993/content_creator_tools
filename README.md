# Content Creator Assistant 

This project is inspired by the [LangChain Academy: Intro to LangGraph course](https://academy.langchain.com/courses/intro-to-langgraph).
The goal is to build an AI-powered assistant that helps generate product review posts.

## 🚀 Project Goal  
The assistant is designed to simulate customer interactions and create meaningful product reviews that can later be used for content creation (e.g., social media posts, product pages, or blogs).  

---

## 🛠️ How It Works  
The workflow involves three main steps:  

1. **Create Customers**  
   - Generate synthetic customers with profiles such as name, occupation, and preferences.  

2. **Ask Questions**  
   - Engage with the customers by asking relevant questions.  
   - Search for their preferred products and compare them to the current product being reviewed.  

3. **Generate Summary**  
   - Use the collected feedback to create a structured product review summarizing the customers’ opinions.  

---
## 🏁 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Tik1993/content_creator_tools.git
cd content_creator_tools
```

### 2. Configure Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
TAVILY_API_KEY=your_search_api_key_here
```

### 3. Run the Project
Run the entry script to start the customer simulation:
```bash
python main.py
```
### 4. Example input
By default, the script in `main.py` runs with this input:
```bash
max_customers = 3
topic = "Write a comment about the latest iphone 17"
```
The assistant will:
1. Generate customer profiles.
2. Ask for your feedback on revisions.
3. Stream updates until a final report is produced.

You’ll be prompted interactively:
```bash
Do you want to revise the customers? (yes/no)
```

## 💡 Notes & Comments  

- This project is a learning exercise inspired by the [LangChain Academy: Intro to LangGraph course](https://academy.langchain.com/courses/intro-to-langgraph).  
- The current implementation focuses on **customer simulation and review generation**, but it can be extended with:  
  - Additional product comparison features  
  - More advanced customer personas  
  - Integration with external data sources for richer context 
