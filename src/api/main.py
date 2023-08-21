"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from api.callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from api.query_data import get_chain, get_chain_RetrievalQA
from api.schemas import ChatResponse

app = FastAPI()
templates_folder = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_folder))
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    # Get the absolute path of the current Python script
    file_path = Path(__file__).resolve()

    # Construct the path to the .env file in the grandparent folder
    env_file_path = file_path.parent.parent / "credentials.env"
    vectorstores_path = file_path.parent \
        .parent.parent / "data" / "vectorstores" / "vectorstore.pkl"

    # Load environment variables from the .env file
    load_dotenv(env_file_path)
    # load_dotenv("credentials.env")
    logging.info("loading vectorstore")
    if not Path(vectorstores_path).exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open(vectorstores_path, "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.websocket("/chat")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     question_handler = QuestionGenCallbackHandler(websocket)
#     stream_handler = StreamingLLMCallbackHandler(websocket)
#     chat_history = []
#     qa_chain = get_chain(vectorstore, question_handler, stream_handler)
#     # Use the below line instead of the above line to enable tracing
#     # Ensure `langchain-server` is running
#     # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

#     while True:
#         try:
#             # Receive and send back the client message
#             question = await websocket.receive_text()
#             resp = ChatResponse(sender="you", message=question, type="stream")
#             await websocket.send_json(resp.dict())

#             # Construct a response
#             start_resp = ChatResponse(sender="bot", message="", type="start")
#             await websocket.send_json(start_resp.dict())

#             result = await qa_chain.acall(
#                 {"question": question, "chat_history": chat_history}
#             )

#             chat_history.append((question, result["answer"]))

#             end_resp = ChatResponse(sender="bot", message="", type="end")
#             await websocket.send_json(end_resp.dict())
#         except WebSocketDisconnect:
#             logging.info("websocket disconnect")
#             break
#         except Exception as e:
#             logging.error(e)
#             resp = ChatResponse(
#                 sender="bot",
#                 message="Sorry, something went wrong. Try again.",
#                 type="error",
#             )
#             await websocket.send_json(resp.dict())

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain_RetrievalQA(vectorstore, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"query": question, "chat_history": chat_history}
            )

            chat_history.append((question, result["result"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
