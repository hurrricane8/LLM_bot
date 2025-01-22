# Telegram Bot for Article Assistance

This repository contains a Telegram bot that helps users navigate and understand the author's articles on optimization and machine learning (https://scholar.google.com/citations?hl=ru&user=dFpnvz4AAAAJ). The bot provides answers to user queries based on the context of the articles using a vector search mechanism and a conversational AI model.

[This bot can be found in telegram](https://t.me/llm_chat_my_test_bot)


## Prerequisites

1. **Python Version**: Ensure you have Python 3.8 or later installed.
2. **Dependencies**: Install the required Python libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

3. **Environment Variables**:
   - Create a `.env` file in the project directory.
   - Add the following environment variables:
     ```env
     TG_BOT_TOKEN=<your_telegram_bot_token>
     OLLAMA_MODEL=<your_ollama_model_name>
     ```
    - You can use OLLAMA_MODEL="llama3.2" as the baseline
4. **Vector Database**:
   - Ensure the FAISS vector database is prepared and stored in `models/faiss_index`.
   - The database should be compatible with the HuggingFace embeddings model `all-MiniLM-L6-v2`.

## Project Structure

- `papers.py`: Script for processing and indexing articles into a FAISS vector database.
- `bot.py`: The main Telegram bot script.
- `requirements.txt`: List of Python dependencies.
- `.env`: File for environment variables (not included in the repository).

## Setup and Usage

### Step 1: Prepare the Vector Database

Run the `papers.py` script to process and index the articles into the FAISS vector database. Ensure all articles are properly formatted and accessible to the script.

```bash
python papers.py
```

### Step 2: Start the Telegram Bot

Once the vector database is ready, run the `bot.py` script to start the Telegram bot. Ensure your bot token and model name are correctly configured in the `.env` file.

```bash
python bot.py
```

### Step 3: Interact with the Bot

1. Open Telegram and start a chat with your bot.
2. Use the `/start` command to begin interacting with the bot.
3. Ask questions related to the articles or use the button to learn more about the author.

## Features

- **Article Context Search**: Retrieves relevant information from the articles using FAISS and HuggingFace embeddings.
- **Conversational Responses**: Generates answers using the specified Ollama model.
- **Interactive Commands**:
  - `/start`: Start the bot and view available options.
  - Button: "Learn about the author" provides information about the project and its creator.

## Notes

- Ensure that `papers.py` is executed before running `bot.py` to ensure the FAISS database is initialized.
- The bot requires an active internet connection to communicate with the AI model and Telegram API.

## Troubleshooting

1. **FAISS Database Not Found**:
   - Ensure the database file `models/faiss_index` exists.
   - Re-run `papers.py` if necessary.

2. **Environment Variables Missing**:
   - Double-check the `.env` file for correct tokens and model names.

3. **Dependencies Not Installed**:
   - Run `pip install -r requirements.txt` to ensure all required libraries are installed.

## License

This project is open-source and available under the MIT License.

