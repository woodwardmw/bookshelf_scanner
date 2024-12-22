from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from api.models import Books, Synopsis
import base64
import json
import asyncio
from openai import AsyncOpenAI  # Example placeholder
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
async_client = AsyncOpenAI()

@app.post("/process-image/")
async def process_image(photo: UploadFile = File(...)):
    print("Starting Processing Image")

    try:
        # Read the image file
        contents = await photo.read()
        if not contents:
            return JSONResponse(content={"success": False, "message": "Empty file received"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)

    # Encode image as base64
    base64_image = base64.b64encode(contents).decode('utf-8')

    # Send to async LLM
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please list the books in this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Books", "schema": Books.model_json_schema()},
        },
    )

    books_data = json.loads(response.choices[0].message.content).get("books", [])
    
    # Process book information in parallel
    results = await asyncio.gather(
        *[fetch_book_info_async(book) for book in books_data],
        return_exceptions=True,
    )
    results = [result for result in results if not isinstance(result, Exception)]
    print(f"Results: {results}")
    return {"success": True, "books": results}

async def fetch_book_info_async(book):
    prompt_template = """
    I want to know more about the following book: "{title}" by {author}.
    Please provide me with the following information:
    1. Book title
    2. A brief description of the book.
    3. A rating of the book, to two decimale places, based on a scale of 1 to 5 (1 being the lowest and 5 being the highest). This should be your estimate of the average rating of the book by readers.
    4. A list of keywords that describe the book.
    5. A list of well-known similar books to "{title}" by {author}.
    """
    prompt = prompt_template.format(title=book["title"], author=book["author"])
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Book_info", "schema": Synopsis.model_json_schema()},
        },
    )
    return json.loads(response.choices[0].message.content)


@app.get("/ping")
async def ping():
    return {"success": True, "message": "Pong!"}
