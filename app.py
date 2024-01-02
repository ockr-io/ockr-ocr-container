import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import socket
import requests
import logging
from inference import ocr
from helper import decode_base64

from pydantic import BaseModel


class OcrRequest(BaseModel):
    parameters: dict = {}
    base64_image: str
    ocr_model_name: str = None
    ocr_model_version: str = None


load_dotenv()
logging.basicConfig(filename='ockr-ocr-app.log',
                    format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
OCKR_REGISTER_ON_STARTUP = os.getenv(
    "OCKR_REGISTER_ON_STARTUP", "true").lower() == "true"
OCKR_MODEL_API_URL = os.getenv(
    "OCKR_API_URL", "http://localhost:9090/api/v1/model/")
OCKR_CONTAINER_PORT = os.getenv('OCKR_CONTAINER_PORT', 5000)
SUPPORTED_OCR_MODELS = ['PP-OCRv3']


@asynccontextmanager
async def lifespan(app: FastAPI):
    if OCKR_REGISTER_ON_STARTUP:
        url = socket.gethostbyname(socket.gethostname())
        registered = {}

        for model in SUPPORTED_OCR_MODELS:
            registered[model] = False

            try:
                requests.post(OCKR_MODEL_API_URL + "register", json={
                              "name": model, "url": url, "port": str(OCKR_CONTAINER_PORT)})
                registered[model] = True
            except Exception as exception:
                logging.error(
                    "Registration of {} failed: {}".format(model, exception))

        yield

        for model in SUPPORTED_OCR_MODELS:
            if registered[model]:
                try:
                    requests.post(OCKR_MODEL_API_URL + "deregister", json={
                                  "name": model, "url": url, "port": str(OCKR_CONTAINER_PORT)})
                except Exception as exception:
                    logging.error(
                        "De-registration of {} failed: {}".format(model, exception))
            else:
                logging.info(
                    "No de-registration for mode {} required, as the registration failed during startup".format(model))

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inference")
def inference(request: OcrRequest):
    image = decode_base64(request.base64_image)
    ocr_model_name = request.ocr_model_name
    ocr_model_version = request.ocr_model_version
    parameters = request.parameters

    if (ocr_model_name == None):
        ocr_model_name = 'PP-OCRv3'

    if (ocr_model_version == None):
        ocr_model_version = 'latest'

    prediction, parameters, actual_model_version = ocr(
        image, ocr_model_name, ocr_model_version, parameters)

    ocr_model_version = actual_model_version
    return {
        "ocr_model_name": ocr_model_name,
        "ocr_model_version": ocr_model_version,
        "parameters": parameters,
        "prediction": prediction
    }


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0",
                port=int(OCKR_CONTAINER_PORT), reload=True)
