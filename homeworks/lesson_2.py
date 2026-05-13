import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o"))

# Потоковый вывод
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты опытный Python разработчик. Ты хорошо понимаешь чужой код и можешь легко"
               "написать docstring к любой функции."),
    ("human", "Напиши docstring на русском языке к функции {code} следующем стиле {style}. В ответе нужна только строка"
              "docstring, без лишних объяснений и без вывода самой функции"),
])

chain = prompt | model

code = """
def calculate_discount(price: float, discount_percent: float) -> float:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
"""
style = "Google"

for chunk in chain.stream({"code": code, 'style': style}):
    print(chunk.content, end="", flush=True)
print()

