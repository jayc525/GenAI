from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

temp1=PromptTemplate(
    template="Write detailed report on {topic}",
    input_variables=["topic"]
)


temp2 = PromptTemplate(
    template='Write a 5 line summary on the following text. {text}',
    input_variables=['text']
)

prompt1 = temp1.invoke({'topic':'black hole'})

result = model.invoke(prompt1)

prompt2 = temp2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)
