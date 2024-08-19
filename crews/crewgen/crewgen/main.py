from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crew import CrewGenCrew
import os

def main():
    load_dotenv()
    # llm=ChatOpenAI(model="gpt-4o", temperature=0.7)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com/beta",
        temperature=0.0
    )

    crew = CrewGenCrew(llm)
    result = crew.run("create note about a given software project or product. research the internet and create a note about the project")
    print("Agent's response:")
    print(result)

if __name__ == "__main__":
    main()