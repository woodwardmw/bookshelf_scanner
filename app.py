import requests

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# origins = ["http://127.0.0.1:8001", "*"]  # Add your frontend origin(s) here
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


# @app.post("/")
# async def main(request: Request, image: UploadFile = File(...)):
#     image_file = await image.read()
#     runner_url = 'https://woodwardmw--run-ratings-predict-process.modal.run'

#     response = requests.post(runner_url, files={'image': image_file})
#     result = response.json()

#     # Render the form template with the request data
#     # form_template = templates.TemplateResponse("form.html", {"request": request})
#     # Render the result template with the request and result data
#     result_template = templates.TemplateResponse("result_page.html", {"request": request, "result": result})

#     # Concatenate the rendered templates into a single response
#     # html = await form_template.render() + await result_template.render()
#     # return HTMLResponse(html)
#     return result_template


@app.post("/process")
async def process_image(image: UploadFile = File(...)):
    print("Starting Processing Image")
    image_file = await image.read()
    runner_url = 'https://woodwardmw--run-ratings-predict-process.modal.run'

    response = requests.post(runner_url, files={'image': image_file})
    results = response.json()
    html = ''
    for i, book in enumerate(results['result_list']):
        print(f'{book=}')
        html += f'''
    <div class="col-md-4 col-lg-3 mb-4">
          <div class="card h-100">
            <a href="{book.get("url")}" target="_blank">
              <img class="card-img-top" src="{book.get("image")}" alt="{book.get("title")}" id="book-image-{i}">
            </a>
            <div class="card-body">
              <h4 class="card-title">{book.get("title")}</h2>
              <p class="card-text">by {book.get("author")}</p>
              <p class="card-text">Rating: {book.get("rating")} | {"{:,}".format(book.get("num_ratings"))} ratings</p>
            </div>
          </div>
        </div>
        '''

    return html