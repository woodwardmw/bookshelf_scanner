import requests

from fastapi import FastAPI, File, UploadFile


app = FastAPI()


@app.get("/")
async def read_root():

    """
    Test docs"""
    return {"Hello": "World"}


@app.post("/image")
async def process_image(image: UploadFile = File(...)):
    image_file = await image.read()
    print(len(image_file))
    runner_url = 'https://woodwardmw--run-ratings-predict-process.modal.run'

    response = requests.post(runner_url, files={'image': image_file})

    return response.json()