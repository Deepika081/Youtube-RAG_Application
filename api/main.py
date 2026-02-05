from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field, field_validator, HttpUrl, AfterValidator
from typing import Annotated
from app.rag_app import Demo

app = FastAPI()
rag_service = Demo()

class RequestValidation(BaseModel):

    url:Annotated[HttpUrl,Field(description="URL of youtube video")]
    query:Annotated[str,Field(description="Query about youtube video")]
    @field_validator('url')
    @classmethod
    def verify_url(cls,url:HttpUrl)->HttpUrl:
        if url.host not in {"youtube.com", "www.youtube.com"}:
            raise ValueError("URL must be a YouTube video URL")
        return url
    
    

@app.post('/query-response')
def query_response(request: RequestValidation):
    try:
        response = rag_service.fetch_response(request.url,request.query)
        return {'message':response}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")