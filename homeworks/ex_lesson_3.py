import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from homeworks.ai_agent import Agent


load_dotenv()


class JobAnalysis(BaseModel):
    position: str = Field(description='Название должности')
    required_skills: List[str] = Field(description='Обязательные навыки')
    nice_to_have_skills: List[str] = Field(description='Желательные навыки')
    experience_years: int = Field(description='Минимальный опыт в годах')
    seniority_level: str = Field(description='Уровень')
    is_remote: bool = Field(description='Удаленная работа')


agent = Agent(
    model_name=os.getenv('MODEL_NAME'),
    base_url=os.getenv('BASE_URL'),
    api_key=os.getenv('API_KEY'),
    temperature=float(os.getenv('MODEL_TEMPERATURE')),
    max_retries=2
)
examples = [
    {
        'input': 'Ищем junior разработчика. Опыт от 1 года.',
        'output': 'Уровень: junior'
    },
    {
        'input': 'Ищем middle разработчика. Опыт от 3 лет.',
        'output': 'Уровень: middle'
    },
    {
        'input': 'Ищем senior разработчика. Опыт от 5 лет.',
        'output': 'Уровень: senior'
    }
]
prompt = agent.prompt(
    role='Ты - специалист по анализу текста вакансий',
    task='Анализировать текст вакансий и находить в тексте ключевые моменты, которые важны для данной вакансии',
    rules="""
        1. Название должности ищи в описании.
        2. Обязательные навыки - это:
        - требования,
        - то что жду от кандидата,
        - то что нужно кандидату
        - ожидания от кандидата.
        3. Желательные навыки - это:
        - то что будет плюсом,
        - то что будет иметь преемущество.
        4. Уровень определяется по ключевым словам:
        - junior: ключевое слово junior или опыт от 1 года,
        - middle: ключевое слово middle или опыт от 3 лет,
        - senior: ключевое слово senior или опыт от 5 лет.
        5. Удаленная работа проставляешь TRUE если есть такие слова как удаленка, удаленная работа, гибридный график
    """,
    examples=examples,
    output_format=JobAnalysis
)

chain = prompt | agent.model | agent.parser
response = chain.invoke({'question': """
    Ищем Python разработчика. Требования: Django, PostgreSQL, 
REST API. Опыт от 3 лет. Знание Docker будет плюсом. 
Офис в Москве, возможна частичная удаленка.
"""})

print(response.position)
print(response.required_skills)
print(response.nice_to_have_skills)
print(response.experience_years)
print(response.seniority_level)
print(response.is_remote)
