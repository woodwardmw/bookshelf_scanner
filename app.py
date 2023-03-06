import requests

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/")
async def process_image(request: Request, image: UploadFile = File(...)):
    image_file = await image.read()
    runner_url = 'https://woodwardmw--run-ratings-predict-process.modal.run'

    response = requests.post(runner_url, files={'image': image_file})
    result = response.json()

    # Render the form template with the request data
    form_template = templates.TemplateResponse("form.html", {"request": request})
    # Render the result template with the request and result data
    result_template = templates.TemplateResponse("result_page.html", {"request": request, "result": result})

    # Concatenate the rendered templates into a single response
    # html = await form_template.render() + await result_template.render()
    # return HTMLResponse(html)
    return result_template