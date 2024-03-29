from langchain.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

class ReviewToolFactory():

    @staticmethod
    def get_review_tool(llm: BaseChatModel) -> Tool:
        """
        Returns a Tool object that can be used for reviewing a result from working on a task and giving constructive 
        feedback. 

        Parameters:
        - llm (BaseChatModel): The language model used for generating the review.

        Returns:
        - Tool: A Tool object that can be used for reviewing a result
        """
        def _review(task: str, result: str) -> str:
            messages = [
                SystemMessage(content="""
                    You are an experienced reviewer with a very broad knowledge. You give constructive feedback. You evalute your criticism step by step.
                    In the end return the review. Alway put youR verdict in the beginning of you review: APPROVED or REJECTED. In the end write REVIEW FINISHED."""),
                HumanMessage(content=f"""
                    Review if the result of the co-worker is correct for the given task. If the result is not good, REJECT it and give contructive feedback.
                    If the result is good you APPROVE it and you may give some hints for further improvement.
                    You are an experienced reviewer with a very broad knowledge. You give constructive feedback. You evalute your criticism step by step.
                    In the end return the review. Alway put yout verdict in the beginning of you review: APPROVED or REJECTED. In the end write REVIEW FINISHED.
                    Task: {task}
                    Result: {result}
                    """),
            ]
            review = llm.invoke(messages)
            print("YYYYYYYY")
            print(review)
            return review.content

        return Tool(
            name="review-tool",
            description="""
            Review a result from working on a task and give constructive feedback. Approves or rejects the result. Parameters:
            - task: the full task that was worked on
            - result: the result that was produced for the task
            """,
            func=_review
        )