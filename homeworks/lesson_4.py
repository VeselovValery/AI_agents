import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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


ai_agent = Agent(
    model_name=os.getenv('MODEL_NAME'),
    base_url=os.getenv('BASE_URL'),
    api_key=os.getenv('API_KEY'),
    temperature=float(os.getenv('MODEL_TEMPERATURE')),
    max_retries=2
)

print(ai_agent.prompt(role='Programmes', rules='GHJKL', output_format=JobAnalysis))


