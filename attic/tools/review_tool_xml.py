from langchain.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pyslurpers import XmlSlurper
from xml.etree import ElementTree
import re
from textwrap import dedent
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
                print(f"ERROR CONTENT-START:{command}CONTENT-END")
                return f"Tool error: Invalid argument. Reason: {e}"

            messages = [
                SystemMessage(content=dedent("""
                    You are an experienced reviewer with a very broad knowledge. You give constructive feedback. You evalute your criticism step by step.
                    In the end return the review. Alway put your verdict in the beginning of you review: APPROVED or REJECTED. In the end write REVIEW FINISHED.""")),
                HumanMessage(content=dedent(f"""
                    Review if the result of the co-worker is correct for the given task. If the result is not good, REJECT it and give contructive feedback.
                    If the result is good you APPROVE it and you may give some hints for further improvement.
                    You are an experienced reviewer with a very broad knowledge. You give constructive feedback. You evalute your criticism step by step.
                    In the end return the review. Alway put yout verdict in the beginning of you review: APPROVED or REJECTED. In the end write REVIEW FINISHED.
                    Topic: {topic}
                    Content: {content}
                    """)),
            ]
            review = llm.invoke(messages)
            return review.content

        return Tool(
            name="review-tool",
            description=dedent("""
            Review a content how good it is matching a given topic or task and give constructive feedback.
            Strictly follow the argument format of the tool in the following XML syntax: 
            <function name="review-tool">
                <topic required="true">$topic</topic>
                <content required="true">$content</content>
            </function> 
            Replace the $variables with actual values. Avoid using XML or HTML tags in the values.        
            The $topic is the topic or task that was worked on and $content is the content that shall be reviewed.
            """),
            func=_review
        )
