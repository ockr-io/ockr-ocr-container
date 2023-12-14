import os
import uvicorn
from fastapi import FastAPI, Response
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import socket
import requests
import logging
import cv2
from inference import ocr

load_dotenv()
logging.basicConfig(filename='ockr-ocr-app.log', format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
OCKR_REGISTER_ON_STARTUP = os.getenv("OCKR_REGISTER_ON_STARTUP", "true").lower() == "true"
OCKR_API_URL = os.getenv("OCKR_API_URL", "http://localhost:8080/api/v1/")
OCKR_CONTAINER_PORT = os.getenv('OCKR_CONTAINER_PORT', 5000)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if OCKR_REGISTER_ON_STARTUP:
        url = socket.gethostbyname(socket.gethostname())
        registered = False

        try:
            requests.post(OCKR_API_URL + "register", json={"name": "ockr-ocr-model", "url": url, "port": str(OCKR_CONTAINER_PORT)})
            registered = True
        except:
            logging.error("Registration failed")
        
        yield
        if registered:
            try:
                requests.post(OCKR_API_URL + "deregister", json={"name": "ockr-ocr-model", "url": url, "port": str(OCKR_CONTAINER_PORT)})
            except:
                logging.error("De-registration failed")
        else:
            logging.info("No de-registration is required as the registration failed during startup")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/inference")
def inference(response: Response):
    image = cv2.imread('test/resources/ockr-specification-abstract-crop.png')
    result = ocr(image, 'PP-OCRv3')

    response.status_code = 200
    return result

if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(OCKR_CONTAINER_PORT), reload=True)
