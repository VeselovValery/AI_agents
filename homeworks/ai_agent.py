import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel


EXAMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ('human', '{input}'),
    ('ai', '{output}'),
])


class Agent:

    def __init__(self, db_history, model_name, base_url, api_key, temperature, max_retries=2):
        self.model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries
        )
        self.db_history = db_history
        self.parser = None

    def get_session_history(self, session_id):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=f'sqlite:///{self.db_history}',
        )

    def trimmer(self, tokens):
        return trim_messages(
            max_tokens=tokens,
            strategy="last",
            token_counter=len,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

    def prompt(
        self,
        role,
        target=None,
        task=None,
        rules=None,
        context=None,
        restrictions=None,
        examples=None,
        output_format=None
    ):
        args = {key: value for key, value in locals().items() if value and key != 'self'}
        system_message = [
            'РОЛЬ: {role}' if role else '',
            'ЦЕЛЬ: {target}' if target else '',
            'ЗАДАЧА: {task}' if task else '',
            'ПРАВИЛА: {rules}' if rules else '',
            'КОНТЕКСТ: {context}' if context else '',
            'ОГРАНИЧЕНИЯ: {restrictions}' if restrictions else ''
        ]
        if isinstance(output_format, type):
            if issubclass(output_format, BaseModel):
                self.parser = PydanticOutputParser(pydantic_object=output_format)
                system_message.append('Ответ выдай в формате JSON: {output_format}')
                args['output_format'] = self.parser.get_format_instructions()
        if examples:
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=EXAMPLE_PROMPT,
                examples=examples,
            )
            prompt_template = ChatPromptTemplate.from_messages([
                ('system', '. '.join(system_message)),
                few_shot_prompt,
                MessagesPlaceholder(variable_name='history'),
                ('human', '{question}'),
            ])
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ('system', '. '.join(system_message)),
                MessagesPlaceholder(variable_name='history'),
                ('human', '{question}'),
            ])
        return prompt_template.partial(**args)

