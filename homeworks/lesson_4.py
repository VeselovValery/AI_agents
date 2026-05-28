import os
import sqlite3

from dotenv import load_dotenv

from homeworks.ai_agent import Agent, AgentPromptConfig


load_dotenv()

agent = Agent(
    model_name=os.getenv('MODEL_NAME'),
    base_url=os.getenv('BASE_URL'),
    api_key=os.getenv('API_KEY'),
    temperature=float(os.getenv('MODEL_TEMPERATURE')),
    db_history='chat_history.db'
)

mode = input('Режим tutor/reviewer (y/n): ')
while mode not in ['y', 'n']:
    mode = input('Режим tutor/reviewer (y/n): ')
if mode == 'y':
    role = 'Терпеливый преподаватель Python, объясняешь концепции с примерами'
else:
    role = 'Строгий code reviewer, указываешь на проблемы и предлагаешь улучшения'
prompt_config = AgentPromptConfig(role=role)
login = input('Логин: ')

chain = agent.chain(prompt_config)


def chat(session_id, message):
    response = chain.invoke(
        {'question': message},
        config={'configurable': {'session_id': session_id}},
    )
    return response.content


def count_messages(session_id, db_path='chat_history.db'):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM message_store WHERE session_id = ?', (session_id,))
        return cursor.fetchone()[0]


while True:
    messages = input(f'{login}: ')
    if messages == '/exit':
        break
    if messages == '/history':
        print(f'Всего сообщений: {count_messages(login)}')
        continue
    print(f'ИИ: {chat(login, messages)}')
