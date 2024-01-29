from langchain.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pyslurpers import XmlSlurper
from xml.etree import ElementTree
class ReviewToolFactory2():

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
        def _review(command: str):
import re

            try:
                # Extract XML content from the command string
                xml_content_match = re.search(r'<function.*?</function>', command, re.DOTALL)
                if not xml_content_match:
                    raise ValueError("No valid XML content found in the command string.")
                xml_content = xml_content_match.group(0)
                xml = XmlSlurper.create(xml_content)
                topic = xml.topic
                content = xml.content
            except (ElementTree.ParseError, ValueError) as e:
                print(f"ERROR CONTENT: {command}END-OF-CONTENT")
                return f"Tool error: command had wrong format. Must be only a valid XML according to the tool specification. Reason: {e.msg}"

            messages = [
                SystemMessage(content="""
                    You are an experienced reviewer with a very broad knowledge. You give constructive feedback. You evalute your criticism step by step.
                    In the end return the review. Alway put your verdict in the beginning of you review: APPROVED or REJECTED. In the end write REVIEW FINISHED."""),
                HumanMessage(content=f"""
                    Review if the result of the co-worker is correct for the given task. If the result is not good, REJECT it and give contructive feedback.
                    If the result is good you APPROVE it and you may give some hints for further improvement.
                    You are an experienced reviewer with a very broad knowledge. You give constructive feedback. You evalute your criticism step by step.
                    In the end return the review. Alway put yout verdict in the beginning of you review: APPROVED or REJECTED. In the end write REVIEW FINISHED.
                    Topic: {topic}
                    Content: {content}
                    """),
            ]
            review = llm.invoke(messages)
            return review.content

        return Tool(
            name="review-tool",
            description="""
            Review a content that was produced as a work product of a given topic or task and give constructive 
            feedback. Approves or rejects the result. Strictly obey the input format of the tool in the following 
            XML syntax. Avoid using XML or HTML tags in the values. 
<function name="review-tool">
	<topic>$topic</topic>
	<content format="string">"$content"</content>
</function>        
            The <topic> is the topic or task that was worked on and <content> is the content that shall be reviewed.
            """,
            func=_review
        )
