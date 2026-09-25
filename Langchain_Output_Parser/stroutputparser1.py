from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
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

parser=StrOutputParser()

chain= temp1 | model | parser | temp2 | model | parser

result=chain.invoke({ "topic":"black hole"})

print(result)