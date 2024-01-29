from pyslurpers import XmlSlurper

data="""<function name="review-tool"> <topic>Personal Agents</topic> <content>The capabilities of Personal Agents are numerous. They can perform various tasks such as work and life assistance, personalized services and recommendations, and proactive assistance. They can also represent users in completing complex affairs, interact with other users or agents, and handle personal data securely. Additionally, Personal LLM Agents have the potential to become a major software paradigm for personal devices in the LLM era, but research is still in its early stages and there are numerous technical challenges to overcome.</content> </function>
observ"""

xml = XmlSlurper.create(data)
print(xml)
# print(xml.location)
# print(xml.date_time)


# xml = XmlSlurper.create("<root><color><name>red</name><rgb>FF0000</rgb></color><color><name>green</name><rgb>00FF00</rgb></color></root>")
# for color in xml.color:
#     print("name: {}, rgb: {}".format(color.name, color.rgb))