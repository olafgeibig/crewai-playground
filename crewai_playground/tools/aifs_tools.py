from aifs import search
from langchain.tools import Tool
from textwrap import dedent
class AifsToolFactory:
    @staticmethod
    def get_search_tool(path: str, max_results: int = 5) -> Tool:
        """
        Create an instance of a semantic search tool that searches documents using a vector DB. The documents are indexed upon the first search.

        Parameters:
        - path (str): The path to the directory to search. Defaults to the current working directory.
        - max_results (int, optional): The maximum number of search results to return. Defaults to 5.
        
        Returns:
        A LangChain tool for searching documents.
        """
        def _search(query: str) -> str:
            result = search(
                query=query,
                path=path,
                max_results=max_results
            )
            return result
        
        return Tool(
            name="aifs-tool",
            description=dedent(
"""A tool that performs a semantic search of documents with valuable information and returns semantically 
relevant excerpts from them. The argument is the query string."""),
            func=_search
        )
