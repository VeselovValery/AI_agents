import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

model = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o"))

# messages = [
#     SystemMessage(content="Ты опытный Python-разработчик. Отвечай кратко и по делу."),
#     HumanMessage(content="Что такое генератор в Python?"),
# ]
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты опытный разработчик. Объясняй технические концепции четко и с примерами кода."),
    ("human", "Объясни концепцию: {concept}"),
])
chain = prompt | model

# response = model.invoke(messages)
response = chain.invoke({"concept": "замыкание в Python"})
print(response.content)
