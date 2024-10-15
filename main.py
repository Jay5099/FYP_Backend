from typing import Union
from fastapi import FastAPI
from queryprocess import queryprocess
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/greet")
async def greet(name: str = "query"):
    return {"message": f"Hello, {name}!"}

@app.get("/querys")
async def querys(query: str = "query"):
    result = queryprocess(query)
    return {"message": f"Query: {result}"}






