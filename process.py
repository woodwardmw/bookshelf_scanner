from pathlib import Path
from typing import List, Dict
import requests
import urllib
from io import BytesIO
import numpy as np
# import cv2
# from roboflow import Roboflow
from PIL import Image
from fastapi import Request
from pydantic import BaseModel
import json 

# from requests_html import HTMLSession
import modal.aio
import asyncio
from starlette.applications import Starlette
from starlette.responses import JSONResponse
import aiohttp
from dotenv import load_dotenv
import openai

load_dotenv('secrets.env')


stub = modal.aio.AioStub(
    "run-ratings-predict",
    image=modal.Image.debian_slim()
    .run_commands("apt update", "apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6")
    .pip_install(
        "roboflow",
        "easyocr",
        "opencv-python==4.1.2.30",
        "requests_html",
        "openai",
        # "read_dot_env",
    )
)

stub.run_ocr = modal.Function.from_name('run-ocr', 'predict')
stub.run_boxes = modal.Function.from_name('run-boxes', 'predict')


class Result(BaseModel):
    title: str
    author: str
    rating: float
    summary: str
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
        self.title = None
        self.text = []
        self.rating = None
        self.num_ratings = None
        self.author = None
        self.image = 'https://d15be2nos83ntc.cloudfront.net/images/no-cover.png'
        self.link = None
        self.summary = None
        self.keep = None
    
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
        self.query = "site:goodreads.com title: " + self.ocr_title + ', author: ' + self.ocr_author if self.ocr_title and self.ocr_author else None
        if not self.query:
            self.link = None
            return None
        print(self.query)
        response = self.get_results()
        self.parse_results(response)
    
    def get_results(self):
        
        self.query_string = urllib.parse.quote_plus(self.query)
        response = get_source("https://www.google.com/search?q=" + self.query_string)
        
        return response

    def parse_results(self, response):
    
        css_identifier_link = ".yuRUbf a"
        css_identifier_title = "h3.LC20lb"
        css_identifier_rating_bar = ".uo4vr"
        css_identifier_num_ratings = ".z3HNkc~ span+ span"
        css_identifier_summary = ".lyLwlc span"
        # css_identifier_text = ".VwiC3b"
        
        # rating = response.html.find(css_identifier_rating, first=True)
        # self.rating = rating.text if rating else None
        # self.link = response.html.absolute_links.pop() if response.html.absolute_links else None
        self.link = response.html.find(css_identifier_link, first=True).attrs['href'] if response.html.find(css_identifier_link, first=True) else None
        self.title = response.html.find(css_identifier_title, first=True).text if response.html.find(css_identifier_title, first=True) else None
        self.rating_bar_text = response.html.find(css_identifier_rating_bar, first=True).text if response.html.find(css_identifier_rating_bar, first=True) else None
        print(self.rating_bar_text)
        try:
            self.rating = float(self.rating_bar_text.split('Rating: ')[-1].split(' · ')[0]) if self.rating_bar_text else None
        except ValueError:
            self.rating = None
        try:
            self.num_ratings = int(self.rating_bar_text.split(' · ‎')[1].split(' votes')[0].replace(',', '')) if self.rating_bar_text else None
        except ValueError:
            self.num_ratings = None
        self.summary = response.html.find(css_identifier_summary, first=True).text if response.html.find(css_identifier_summary, first=True) else None
        self.summary = self.summary.split('largest community for readers. ')[-1] if self.summary else None

    def goodreads_scrape(self):
        if self.link is None:
            return None
        else:
            print(f"Scraping Goodreads for {self.link}")
            response = get_source(self.link)
            # css_identifier_title = ".Text__title1"
            # css_identifier_rating = ".RatingStatistics__rating"
            # css_identifier_authors = ".ContributorLink__name"
            # css_identifier_num_ratings = ".RatingStatistics__meta"
            css_identifier_image = ".BookCover__image img"
            # print(f'Returned HTML for {self.link}: {response.html.html}')
            # title = response.html.find(css_identifier_title, first=True)
            # self.title = title.text if title else None
            # rating = response.html.find(css_identifier_rating, first=True)
            # num_ratings = response.html.find(css_identifier_num_ratings, first=True)
            # self.num_ratings = int(num_ratings.text.split()[0].replace(',', '')) if num_ratings else None
            # authors = response.html.find(css_identifier_authors)
            # for author in authors:
            #     if author.text not in self.authors:
            #         self.authors.append(author.text)
            # self.rating = rating.text if rating else None
            self.image = response.html.find(css_identifier_image, first=True).attrs['src'] if response.html.find(css_identifier_image, first=True) else None
            print(self.image)
            # print(self.rating)
            # print(self.num_ratings)
            # print(self.authors)
            # print(self.title)
    def __repr__(self):
        return f"""
        BookBox(
            {self.x1}, {self.y1}, {self.x2}, {self.y2}
            {self.text}
            {self.title}
            {self.author}
            {self.rating}
            {self.num_ratings}
            {self.image}
            {self.link}
            {self.summary}
            {self.keep}
            )
        """


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


@stub.function()
async def get_text(image):
    results = modal.container_app.run_ocr.call(image)
    return results


@stub.function()
async def get_boxes(image):
    results = modal.container_app.run_boxes.call(image)

    return results


@stub.function()
async def attach_text_to_book_box(text_boxes, book_box):
    print('Attaching text to book box...')
    for text_box in text_boxes:
        if book_box.get_overlap(text_box) > 0.5:
            book_box.text.append(text_box.text)
    return book_box


@stub.function()
async def get_online_data(text_boxes, book_box):
    print('Getting Google results...')
    book_box.google_search()
    print('Getting Goodreads results...')
    book_box.goodreads_scrape()
    return book_box


def parse_book_info(book_info: list):
    book_info_output = []
    for book in book_info:
        book_info_output.append({
        'title': book['title'].title(),
        'author': book['author'].title(),
        # 'summary': book['summary'],
        # 'rating': book['rating'],
    })
    return book_info_output

@stub.function(secret=modal.Secret.from_name("openai"))
def chatgpt_cleanup(book_text: str):
    import json
    # import openai
    # import os
    # openai.api_key = os.getenv('OPENAI_API_KEY')
    # print(openai.api_key)
    prompt = f"""
    You are a master-level librarian who knows every book in the world, and their goodreads summary and rating.
    I will give you a list of arrays, with each array containing a list of words.
    For each array, you need to rearrange the words to make a title and author of a real-life book.
    Work one step at a time.
    Next, you need to summarize the book in one paragraph, outlining the main themes and ideas.
    Finally, add a rating from 1-5, with 5 being the best. The rating should be a number, and should be the rating on goodreads.com.

    Include all books that you can confidently identify from the OCR. If you cannot identify a book from an array, return an empty dictionary for that book.
    
    The OCR output is quoted in the backticks:
    ```
    {book_text}
    ```
    Respond with only the json, and no other comments.
    The output should be json, in the following format:
    [{{title: <title>,
    author: <author>,
    summary: <summary>,
    rating: <rating>}}, ...]
    For example:
    Input: [['Gatsby', 'The', '14', '&', 'F._Sc0tt', 'Great', '8', 'FitzGRErald'], ...]
    Output: [{{
        "title": "The Great Gatsby", 
        "author": "F. Scott Fitzgerald", 
        "summary": "A classic American novel that explores the decadence and disillusionment of the Jazz Age through the eyes of Jay Gatsby, a mysterious millionaire. It is beloved for its vivid portrayal of the Roaring Twenties and its examination of the American Dream's elusive nature.", 
        "rating": 3.91
        }}, ...]
    """
    messages = [{"role": "user", "content": prompt}]

    functions = [
        {
            "name": "parse_book_info",
            "description": "Parse information about a book",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_info": {
                        "type": "array",
                        "description": "A list of dicts containing information about a book",
                        "items": {
                            "type": "object",
                            "properties": {
                                'title': {'type': 'string'}, 
                                'author': {'type': 'string'}, 
                                # 'summary': {'type': 'string'}, 
                                # 'rating': {'type': 'number'}
                            },
                        },
                    },
                },
                "required": ["book_info"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-0613",
                            # functions=functions,
                            temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
    response_message = response['choices'][0]["message"]
    print(f'{response_message=}')

    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "parse_book_info": parse_book_info,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            book_info=function_args.get("book_info"),
        )
        print(f'{function_response=}')

        # Step 4: send the info on the function call and function response to GPT
        # messages.append(response_message)  # extend conversation with assistant's reply
        # messages.append(
        #     {
        #         "role": "function",
        #         "name": function_name,
        #         "content": function_response,
        #     }
        # )  # extend conversation with function response
        # print("Running second function")
        # second_response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo-0613",
        #     messages=messages,
        # )  # get a new response from GPT where it can see the function response
        # return second_response
        return function_response

    return response_message['content']


def calculate_word_overlap(string1, string2):
    words1 = string1.lower().split(':')[0].split()
    words2 = string2.lower().split(':')[0].split()

    num_same_words = 0
    for word1 in words1:
        for word2 in words2:
            if calculate_word_similarity(word1, word2) >= 0.8:
                num_same_words += 1
                break

    overlap = num_same_words / min(len(words1), len(words2))
    
    return overlap >= 0.3


def calculate_word_similarity(word1, word2):
    letters1 = set(word1.lower())
    letters2 = set(word2.lower())
    common_letters = letters1.intersection(letters2)
    
    similarity = len(common_letters) / max(len(letters1), len(letters2))
    return similarity


@stub.function()
@modal.web_endpoint(method="POST")
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
    for book_box in text_results:
        text_boxes.append(TextBox(book_box[0][0], book_box[0][2], book_box[1], book_box[2]))
    
    # for text_box in text_boxes:
    #     text_box.text = book_texts.pop(0)
    
    for book_box in box_results['predictions']:
        book_boxes.append(
                        BookBox(
                            [int(book_box['x'] - 0.5*book_box['width']), int(book_box['y'] - 0.5*book_box['height'])], 
                            [int(book_box['x'] + 0.5*book_box['width']), int(book_box['y'] + 0.5*book_box['height'])]
                                )
                        )
    print("Text boxes and book boxes created")
    coroutines = [attach_text_to_book_box.call(text_boxes, book_box) for book_box in book_boxes]
    book_boxes = await asyncio.gather(*coroutines)
    book_texts = [book_box.text for book_box in book_boxes]
    print(f'{book_texts=}')
    book_texts = await chatgpt_cleanup.call(book_texts)
    book_texts = json.loads(book_texts)
    print(f'{book_texts=}')

    for i, book_box in enumerate(book_boxes):  # If there are blank books, there may be an issue here with it lining up with the wrong box, so that may need fixing
        book_box.ocr_title = book_texts[i].get('title') if len(book_texts) > i else None
        book_box.ocr_author = book_texts[i].get('author') if len(book_texts) > i else None
        book_box.summary = book_texts[i].get('summary') if len(book_texts) > i else None
        book_box.rating = book_texts[i].get('rating') if len(book_texts) > i else None

    coroutines = [get_online_data.call(text_boxes, book_box) for book_box in book_boxes]
    book_boxes_with_online = await asyncio.gather(*coroutines)
    for i, book_box in enumerate(book_boxes_with_online):
        if book_box.title is None or book_boxes[i].ocr_title is None:
            book_box.keep = False
        else:
            book_box.keep = calculate_word_overlap(book_box.title, book_boxes[i].ocr_title)
        print(book_box)

    # await get_google_data.call(text_boxes, book_boxes)
    print("Online data received")
    book_boxes_with_online = sorted([result for result in book_boxes_with_online if result.rating] , key=lambda x: x.rating, reverse=True)
    result_list = []
    for book_box in book_boxes_with_online:
        if book_box.keep:
            result_list.append(Result(
                title=book_box.title if book_box.title else 'No title found',
                rating=book_box.rating if book_box.rating else 0.0,
                num_ratings=book_box.num_ratings if book_box.num_ratings else 0,
                author=book_box.ocr_author if book_box.ocr_author else ['No authors found'],
                summary=book_box.summary if book_box.summary else 'No summary found',
                image=book_box.image if book_box.image else 'https://d15be2nos83ntc.cloudfront.net/images/no-cover.png',
                url=book_box.link if book_box.link else 'https://www.goodreads.com/'
            ))
            print(f'{book_box.title} - {book_box.rating}')

    return {'result_list': result_list}


if __name__ == "__main__":
    asyncio.run(process())
