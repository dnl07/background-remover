import uvicorn
from fastapi import FastAPI

app = FastAPI()

app.post("inference/")
async def inference():
    pass

def run():
    uvicorn.run("main:app", host="")