import dataclasses

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import trim_messages, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel


EXAMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ('human', '{input}'),
    ('ai', '{output}'),
])


@dataclasses.dataclass
class AgentPromptConfig:
    role: str
    target: str | None = None
    task: str | None = None
    rules: str | None = None
    context: str | None = None
    restrictions: str | None = None
    examples: list | None = None
    output_format: type | None = None


class Agent:

    def __init__(self, db_history, model_name, base_url, api_key, temperature, max_retries=2, history_limit_tokens=20):
        self.model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries
        )
        self.db_history = db_history
        self.history_limit_tokens = history_limit_tokens
        self.parser = None

    def get_session_history(self, session_id):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=f'sqlite:///{self.db_history}',
        )

    def trimmer(self):
        return trim_messages(
            max_tokens=self.history_limit_tokens,
            strategy='last',
            token_counter=len,
            include_system=True,
            allow_partial=False,
            start_on='human',
        )

    def prompt(self, config: AgentPromptConfig):
        args = {key: value for key, value in dataclasses.asdict(config).items() if value}
        system_message = [
            'РОЛЬ: {role}' if config.role else '',
            'ЦЕЛЬ: {target}' if config.target else '',
            'ЗАДАЧА: {task}' if config.task else '',
            'ПРАВИЛА: {rules}' if config.rules else '',
            'КОНТЕКСТ: {context}' if config.context else '',
            'ОГРАНИЧЕНИЯ: {restrictions}' if config.restrictions else ''
        ]
        if isinstance(config.output_format, type):
            if issubclass(config.output_format, BaseModel):
                self.parser = PydanticOutputParser(pydantic_object=config.output_format)
                system_message.append('Ответ выдай в формате JSON: {output_format}')
                args['output_format'] = self.parser.get_format_instructions()
        if config.examples:
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=EXAMPLE_PROMPT,
                examples=config.examples,
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

    def chain(self, config: AgentPromptConfig):
        prompt = self.prompt(config)
        if self.parser:
            chain = (
                RunnablePassthrough.assign(history=RunnableLambda(lambda x: self.trimmer().invoke(x["history"])))
                | prompt
                | self.model
                | self.parser
                | RunnableLambda(lambda x: AIMessage(content=x.model_dump_json()))
            )
        else:
            chain = (
                RunnablePassthrough.assign(history=RunnableLambda(lambda x: self.trimmer().invoke(x["history"])))
                | prompt
                | self.model
            )
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key='question',
            history_messages_key='history',
        )

