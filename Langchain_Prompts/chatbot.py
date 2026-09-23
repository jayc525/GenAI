from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

chat_history=[]

while True:
    text = input('You: ')
    if text == 'exit':
        break
    chat_history.append(HumanMessage(content=text))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    if isinstance(response.content, list):
        print("Bot: ", response.content[0]["text"])
    else:
        print("Bot: ", response.content)