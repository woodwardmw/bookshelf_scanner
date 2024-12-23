from typing import List, Optional
from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: Optional[str] = None

class Books(BaseModel):
    books: List[Book]

class Synopsis(BaseModel):
    title: str
    author: Optional[str] # Author or authors of the book
    rating: float = 0.00
    description: Optional[str] = None
    keywords: List[str] = []
    similar_books: List[Book] = []
