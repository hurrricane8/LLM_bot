import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from ollama import chat
from dotenv import load_dotenv

load_dotenv()

TG_BOT_TOKEN = os.getenv('TG_BOT_TOKEN')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')

PROMPT_TEMPLATE = """
Используя следующий контекст, ответь на вопрос. Если ответа в контексте нет, скажи об этом явно.

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""

# Загрузка векторной базы данных
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("models/faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def query_ollama(prompt):
    try:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': 'Ты являешься телеграм ботом - помощником, который помогает людям разобраться в статьях автора по оптимизации и машинном обучении.'},
                {'role': 'user', 'content': prompt},
            ],
        )
        return response.message.content
    except Exception as e:
        return f'Ошибка: {e}'


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    await update.message.reply_text("Разбираюсь в вопросе и анализирую статьи, мне нужно немного времени")

    try:
        retrieved_docs = retriever._get_relevant_documents(user_message, run_manager=None)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = PROMPT_TEMPLATE.format_map({
            'context': context,
            'question': user_message})
        response = query_ollama(prompt)

        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text('Произошла ошибка, попробуйте снова.')


async def about_author(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик для кнопки "Узнать информацию об авторе"."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(
        text="Этот бот был разработан в рамках курса по машинному обучению. Автор — студент, который пишет научные статьи и хочет, чтобы другие могли лучше и быстрее усваивать материал!"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    keyboard = [
        [InlineKeyboardButton("Узнать информацию об авторе", callback_data='about_author')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        '👋 Привет! Я бот-путеводитель по статьям. Здесь вы можете задать любой вопрос по статьям автора. Нажмите на кнопку ниже, чтобы узнать больше обо мне, или напишите свой вопрос!',
        reply_markup=reply_markup
    )


def main():
    app = ApplicationBuilder().token(TG_BOT_TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CallbackQueryHandler(about_author, pattern='about_author'))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print('Бот запущен и готов к работе...')
    app.run_polling()


if __name__ == '__main__':
    main()
