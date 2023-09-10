# 🦜️🔗 ChatLangChain

This repo is an implementation of a locally hosted chatbot specifically focused on question answering over the [LangChain documentation](https://langchain.readthedocs.io/en/latest/). It is ingested with information about the bitcoin law in El Salvador for demostration purposes.
Built with [LangChain](https://github.com/hwchase17/langchain/) and [FastAPI](https://fastapi.tiangolo.com/).

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## ✅ Credentials
1. Remeber to get a API-key from openai and save it in credentials.env

## ✅ Running with container
1. docker compose --env-file credentials.env up

## ✅ Running locally
1. Install dependencies: `pip install -r requirements.txt`
1. Verify that you have the openai API-KEY in credentials.env file
1. Run the app: `uvicorn src.api.main:app --reload --port 9001`
1. Open [localhost:9001](http://localhost:9001) in your browser.