import os
import re
import sqlite3
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import mimetypes
from typing import List

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize GenAI client
client = genai.Client(api_key=API_KEY)
model = "gemini-2.5-flash-image-preview"

# FastAPI app
app = FastAPI()

# Mount static files for serving generated images
os.makedirs("./images", exist_ok=True)
app.mount("/images", StaticFiles(directory="./images"), name="images")

# SQLite DB setup
DB_FILE = "generations.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS generations (
        id TEXT PRIMARY KEY,
        prompt TEXT,
        dress_filename TEXT,
        model_filename TEXT,
        generated_filename TEXT,
        timestamp DATETIME
    )
"""
)
conn.commit()


# Pydantic model for request validation
class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=10,
        description="Prompt must be a string with at least 10 characters",
    )

    @classmethod
    def prompt_must_be_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("Prompt must be a string")
        # Remove leading/trailing whitespace and collapse multiple spaces
        v = re.sub(r"\s+", " ", v.strip())
        if len(v) < 10:
            raise ValueError(
                "Prompt must be at least 10 characters long after removing extra spaces"
            )
        return v


# Pydantic model for response
class GenerateResponse(BaseModel):
    image_url: str
    generation_id: str
    text_output: str | None = None


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(
    images: List[UploadFile] = UploadFile(...), prompt: str = Form(...)
):
    try:
        # Validate number of images (at least 2)
        if len(images) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two images are required (e.g., dress and model)",
            )

        # Validate prompt using Pydantic
        try:
            cleaned_prompt = GenerateRequest.prompt_must_be_string(prompt)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Read and validate uploaded images
        image_parts = []
        filenames = []
        for i, image in enumerate(images):
            image_bytes = await image.read()
            mime_type = mimetypes.guess_type(image.filename)[0] or (
                "image/jpeg" if image.filename.lower().endswith(".jpg") else "image/png"
            )
            if not mime_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file type for {image.filename}",
                )
            image_parts.append(
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            )
            filenames.append(image.filename)

        # Define contents (prompt + images)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=cleaned_prompt),
                    *image_parts,  # Spread all image parts
                ],
            )
        ]

        # Generate content config (dictionary-based)
        generate_content_config = {
            "response_modalities": ["TEXT", "IMAGE"],
            "max_output_tokens": 200,
        }

        # Generate content
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        # Extract text output (if any)
        text_output = response.text if response.text else None

        # Extract image parts
        image_parts = [
            part.inline_data.data
            for candidate in response.candidates
            for part in candidate.content.parts
            if part.inline_data and part.inline_data.data
        ]

        if not image_parts:
            raise HTTPException(status_code=500, detail="No image generated")

        # Save the first generated image (assuming one for simplicity)
        generated_image = Image.open(BytesIO(image_parts[0]))
        generated_filename = f"generated_{uuid.uuid4().hex}.png"
        generated_path = os.path.join("./images", generated_filename)
        generated_image.save(generated_path)

        # Store in DB
        generation_id = uuid.uuid4().hex
        timestamp = datetime.now().isoformat()
        dress_filename = filenames[0] if len(filenames) > 0 else "unknown"
        model_filename = filenames[1] if len(filenames) > 1 else "unknown"
        cursor.execute(
            """
            INSERT INTO generations (id, prompt, dress_filename, model_filename, generated_filename, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                generation_id,
                cleaned_prompt,
                dress_filename,
                model_filename,
                generated_filename,
                timestamp,
            ),
        )
        conn.commit()

        # Return clickable URL (localhost:8000/images/generated_xxx.png)
        image_url = f"http://localhost:8000/images/{generated_filename}"

        return GenerateResponse(
            image_url=image_url, generation_id=generation_id, text_output=text_output
        )

    except genai.errors.APIError as e:
        raise HTTPException(
            status_code=500, detail=f"GenAI API Error: {e.code} - {e.message}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# Run the app (for development: uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
