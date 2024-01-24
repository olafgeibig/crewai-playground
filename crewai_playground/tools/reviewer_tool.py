from langchain.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
class ReviewerToolFactory():

    def get_reviewer_tool(reviewer_llm: BaseChatModel) -> Tool:
        
        def _review(command: str) -> str:
            try:
                task, content = command.split("|")
            except ValueError:
                return "Tool error: command had wrong format. Must be <task>|<result>"
            review = reviewer_llm.generate(
            """
            Review if the result of the coworker is correct for the given task. If the result is not good, give good and detailed reasons.
            You are an experienced reviewer with a very broad knowledge. You give constructive feedback. You evalute your criticism step by step.
            Task: {task}
            Result: {result}
            """)
            return review

        return Tool(
            name="Review work results and give constructive feedback",
            description="""
            Reviews work results thorougly if it is a good sresponse to the task. Decides if the task has been successfully fulfilled.
            Give valuable feedback that helps to improve your results.
            Strictly obey the input format of the tool argument: task|result. The task is the task that was worked on and result is your final result.
            task and result must be seperated by a pipe sign | and concatenated.
            """,
            func=_review
        )            