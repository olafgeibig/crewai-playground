from langchain_community.tools import BraveSearch
import os
from dotenv import load_dotenv

load_dotenv()
search_tool = BraveSearch.from_api_key(api_key=os.getenv('BRAVE_API_KEY'), search_kwargs={"count": 5})

