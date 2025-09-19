from fastapi import FastAPI

app = FastAPI(title="Mock MongoDB API")

@app.get("/get_candidate_id")
async def get_candidate_id(user_id: str):
    return {"candidate_id": f"{user_id}"}