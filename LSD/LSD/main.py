from fastapi import FastAPI
from LSD import model
from LSD.database import engine, Base
from LSD.routers import sqli, user, history, chatbot
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
model.Base.metadata.create_all(bind=engine)
app.include_router(user.router)
app.include_router(sqli.router)
app.include_router(chatbot.router)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
