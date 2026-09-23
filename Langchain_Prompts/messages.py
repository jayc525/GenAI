from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")    

messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="tell me about langchain")
    # AIMessage(content="Hello! How can I help you today?")
]

result= model.invoke(messages)

messages.append(AIMessage(content=result.content))


print(messages)
