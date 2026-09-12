from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from llada_model import run_generate  # loads the model once on import

app = FastAPI()

class GenRequest(BaseModel):
    prompt: str
    gen_length: int = 256
    block_length: int = 32

@app.post("/generate")
def generate(req: GenRequest):
    text = run_generate(req.prompt, req.gen_length, req.block_length)
    return {"response": text}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)