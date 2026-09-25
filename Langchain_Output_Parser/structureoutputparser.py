from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=1000
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(
        name="fact_1",
        description="First fact about the topic"
    ),
    ResponseSchema(
        name="fact_2",
        description="Second fact about the topic"
    ),
    ResponseSchema(
        name="fact_3",
        description="Third fact about the topic"
    )
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="""
Give 3 facts about {topic}.

{format_instruction}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({
    "topic": "black hole"
})

print(result)