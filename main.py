from fastapi import FastAPI
from config.config import TrainPayLoad, TestPayLoad
from model.train import train, make_predictions


def add_numbers():
    return 

app = FastAPI()


@app.get("/")
def read_root():
    return {"Msg": "Hello World!", "status":"The App is running fine"}


@app.post("/train")
def train_model(payload: TrainPayLoad):
    resp = train(payload)
    return {"status":"success", "msg":f"{resp}"}


@app.post("/predict")
def predict(payload: TestPayLoad):
    resp = make_predictions(payload)
    return {"status":"success", "msg":f"{resp}"}