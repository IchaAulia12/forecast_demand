# backend.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from predictor import DemandPredictor
import uvicorn

app = FastAPI()
predictor = DemandPredictor(model_path="artifacts/demand_forecast.pkl",
                            cols_path="artifacts/model_cols.pkl",
                            gm_path="artifacts/global_means.pkl")

class HistoryRow(BaseModel):
    date: str  # yyyy-mm-dd
    store: int
    item: int
    sales: float

class ForecastRequest(BaseModel):
    store: int
    item: int
    steps: int

@app.post("/add_history")
def add_history(row: HistoryRow):
    # append to local CSV (simple)
    df_row = pd.DataFrame([row.dict()])
    try:
        df_row.to_csv("user_history.csv", mode='a', header=not pd.io.common.file_exists("user_history.csv"), index=False)
        return {"status":"ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast")
def forecast(req: ForecastRequest):
    # read history
    try:
        df_hist = pd.read_csv("user_history.csv", parse_dates=['date'])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="history not found")
    try:
        preds = predictor.multi_step_forecast(df_hist, req.store, req.item, steps=req.steps)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# run with: uvicorn backend:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
