from pathlib import Path
from typing import List
import requests
import urllib
from io import BytesIO
import numpy as np
# import cv2
# from roboflow import Roboflow
from PIL import Image
from fastapi import Request
from pydantic import BaseModel

# from requests_html import HTMLSession
import modal.aio
import asyncio
from starlette.applications import Starlette
from starlette.responses import JSONResponse
import aiohttp



stub = modal.aio.AioStub(
    "run-ratings-predict",
    image=modal.Image.debian_slim()
    .run_commands("apt update", "apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6")
    .pip_install(
        "roboflow",
        "easyocr",
        "opencv-python==4.1.2.30",
        "requests_html",
    )
)

stub.run_ocr = modal.Function.from_name('run-ocr', 'predict')
stub.run_boxes = modal.Function.from_name('run-boxes', 'predict')


class Result(BaseModel):
    title: str
    authors: list
    rating: float
    num_ratings: int
    image: str
    url: str


def get_source(url):
    """Return the source code for the provided URL. 

    Args: 
        url (string): URL of the page to scrape.

    Returns:
        response (object): HTTP response object from requests_html. 
    """
    from requests_html import HTMLSession

    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)


class BookBox:
    def __init__(self, top_left: List[int], bottom_right: List[int]) -> None:
        self.x1 = top_left[0]
        self.y1 = top_left[1]
        self.x2 = bottom_right[0]
        self.y2 = bottom_right[1]
        self.text = []
        self.rating = None
        self.num_ratings = None
        self.authors = []
    
    def get_overlap(self, text_box):

        # Find the overlap in the x direction
        overlap_x = max(0, min(self.x2, text_box.x2) - max(self.x1, text_box.x1))
        # Find the overlap in the y direction
        overlap_y = max(0, min(self.y2, text_box.y2) - max(self.y1, text_box.y1))
        # Calculate the total overlapping area
        overlap_area = overlap_x * overlap_y
        # Calculate the total area of the second rectangle
        rect2_area = (text_box.x2 - text_box.x1) * (text_box.y2 - text_box.y1)
        # Calculate the percentage overlap
        overlap = overlap_area / rect2_area
        return overlap
    
    def google_search(self):
        self.query = "site:goodreads.com " + " ".join(self.text)
        print(self.query)
        response = self.get_results()
        self.parse_results(response)
    
    def get_results(self):
        
        self.query_string = urllib.parse.quote_plus(self.query)
        response = get_source("https://www.google.com/search?q=" + self.query_string)
        
        return response

    def parse_results(self, response):
    
        css_identifier_link = ".yuRUbf a"
        # css_identifier_text = ".VwiC3b"
        
        # rating = response.html.find(css_identifier_rating, first=True)
        # self.rating = rating.text if rating else None
        # self.link = response.html.absolute_links.pop() if response.html.absolute_links else None
        self.link = response.html.find(css_identifier_link, first=True).attrs['href'] if response.html.find(css_identifier_link, first=True) else None

    def goodreads_scrape(self):
        if self.link is None:
            return None
        else:
            response = get_source(self.link)
            css_identifier_title = ".Text__title1"
            css_identifier_rating = ".RatingStatistics__rating"
            css_identifier_authors = ".ContributorLink__name"
            css_identifier_num_ratings = ".RatingStatistics__meta"
            css_identifier_image = ".BookCover__image img"
            title = response.html.find(css_identifier_title, first=True)
            self.title = title.text if title else None
            rating = response.html.find(css_identifier_rating, first=True)
            num_ratings = response.html.find(css_identifier_num_ratings, first=True)
            self.num_ratings = int(num_ratings.text.split()[0].replace(',', '')) if num_ratings else None
            authors = response.html.find(css_identifier_authors)
            for author in authors:
                if author.text not in self.authors:
                    self.authors.append(author.text)
            self.rating = rating.text if rating else None
            self.image = response.html.find(css_identifier_image, first=True).attrs['src'] if response.html.find(css_identifier_image, first=True) else None
            print(self.image)
    def __repr__(self):
        return f"BookBox({self.x1}, {self.y1}, {self.x2}, {self.y2}, {self.text})"


class TextBox:
    def __init__(self, top_left: List[int], bottom_right: List[int], text: str, score: float ) -> None:
        self.x1 = top_left[0]
        self.y1 = top_left[1]
        self.x2 = bottom_right[0]
        self.y2 = bottom_right[1]
        self.text = text
        self.score = score
    # print out TextBox object
    def __repr__(self):
        return f"TextBox({self.x1}, {self.y1}, {self.x2}, {self.y2}, {self.text}, {self.score})"


@stub.function
async def get_text(image):
    results = modal.container_app.run_ocr.call(image)
    return results


@stub.function
async def get_boxes(image):
    results = modal.container_app.run_boxes.call(image)

    return results


@stub.function
async def get_online_data(text_boxes, book_box):
    for text_box in text_boxes:
        if book_box.get_overlap(text_box) > 0.75:
            book_box.text.append(text_box.text)

    # print(book_box)
    print('Getting Google results...')
    book_box.google_search()
    print('Getting Goodreads results...')
    book_box.goodreads_scrape()
    return book_box


@stub.webhook(method="POST")
async def process(request: Request):
    async with aiohttp.ClientSession():
        data = await request.form()
        if 'image' not in data:
            return JSONResponse({'error': 'No image received'})
        image = await data['image'].read()
    
    print("Starting image processing...")
    import cv2
    image_bytes = BytesIO(image)
    print("Image loaded")

    image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    print("Image decoded")
    task1 = asyncio.create_task(get_text.call(image))
    task2 = asyncio.create_task(get_boxes.call(image))
    text_results, box_results = await asyncio.gather(task1, task2)
    print("Text and boxes received")
    text_boxes = []
    book_boxes = []
    for result in text_results:
        text_boxes.append(TextBox(result[0][0], result[0][2], result[1], result[2]))
    for result in box_results['predictions']:
        book_boxes.append(
                        BookBox(
                            [int(result['x'] - 0.5*result['width']), int(result['y'] - 0.5*result['height'])], 
                            [int(result['x'] + 0.5*result['width']), int(result['y'] + 0.5*result['height'])]
                                )
                        )
    print("Text boxes and book boxes created")

    coroutines = [get_online_data.call(text_boxes, book_box) for book_box in book_boxes]
    results = await asyncio.gather(*coroutines)
    print(results)
    # await get_google_data.call(text_boxes, book_boxes)
    print("Online data received")
    results = sorted([result for result in results if result.rating] , key=lambda x: x.rating, reverse=True)
    result_list = []
    for result in results:
        result_list.append(Result(
            title=result.title if result.title else 'No title found',
            rating=result.rating if result.rating else 0.0,
            num_ratings=result.num_ratings if result.num_ratings else 0,
            authors=result.authors if result.authors else ['No authors found'],
            image=result.image if result.image else 'https://www.goodreads.com/assets/nocover/111x148.png',
            url=result.link if result.link else 'https://www.goodreads.com/'
        ))
        print(f'{result.title} - {result.rating}')

    return {'result_list': result_list}


if __name__ == "__main__":
    asyncio.run(process())
