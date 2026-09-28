from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url="https://scienceblog.com/k-two-groups-of-adults-were-taught-the-same-set-of-invented-characters-one-by-copying-them-out-by-hand-and-one-by-typing-them-when-they-were-later-shown-those-characters-and-asked-only-to-recognise-th/"

loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'Summarize the following text in one paragraph.', 'text':docs[0].page_content})) 