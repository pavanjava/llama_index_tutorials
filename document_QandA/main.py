from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.chat_engine.types import ChatMode
from pydantic import BaseModel
from src.core import load_gpt35_turbo

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    input: str


@app.get("/health")
def health():
    return {'health': 'OK'}


@app.post("/chat")
def chat(payload: Payload):
    query_engine = load_gpt35_turbo().as_chat_engine(verbose=False, chat_mode=ChatMode.REACT)
    print(payload.input)
    result = query_engine.chat(payload.input)
    return {'data': str(result)}
