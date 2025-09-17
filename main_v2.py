import os
import sqlite3
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Note: We don't need PIL/Pillow for saving base64 data directly.
# from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import mimetypes
from typing import List
import re
import base64  # Needed to decode base64 image data

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# --- Initialize GenAI client (ASYNC) ---
# Use the asynchronous client via client.aio
client = genai.Client(api_key=API_KEY)
model = "gemini-2.5-flash-image-preview"  # Ensure this model supports image generation

# Metadata for Swagger UI
tags_metadata = [
    {
        "name": "Image Generation",
        "description": "Generate professional e-commerce fashion photos by combining input images based on a prompt.",
    },
]

# FastAPI app
app = FastAPI(
    title="Fashion Image Generation API",
    description="API to generate professional e-commerce fashion photos by combining input images based on a prompt.",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# Mount static files for serving generated images
os.makedirs("./images", exist_ok=True)
app.mount("/images", StaticFiles(directory="./images"), name="images")

# SQLite DB setup (Remains synchronous)
DB_FILE = "generations.db"
# check_same_thread=False allows usage in async context, but be cautious of blocking.
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS generations (
        id TEXT PRIMARY KEY,
        prompt TEXT,
        input_filenames TEXT,
        generated_filename TEXT,
        timestamp DATETIME
    )
    """
)

# Migrate schema if needed
cursor.execute("PRAGMA table_info(generations)")
columns = {row[1] for row in cursor.fetchall()}  # Set of column names
if "text_output" not in columns:
    # cursor.execute("ALTER TABLE generations ADD COLUMN input_filenames TEXT")
    cursor.execute("ALTER TABLE generations ADD COLUMN text_output TEXT")
    print("Added 'input_filenames' column to generations table")
conn.commit()


# Pydantic model for request validation
class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=10,
        description="Prompt must be a string with at least 10 characters after whitespace sanitization",
    )

    @classmethod
    def prompt_must_be_string(
        cls, v
    ):  # This method name might be confusing as a validator
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
    text_output: str | None = None  # Optional text output from the model


API_DESCRIPTION = """
# Best Practices for Crafting Prompts:
- **Use Clear, Descriptive Language**: Be specific about colors, styles, settings, and moods (e.g., "a sleek black leather jacket in a modern urban setting" instead of "a cool jacket").
- **Be Hyper-Specific**: Provide detailed descriptions (e.g., "ornate elian plate armor, etched with silver leaf patterns" instead of "fantasy armor").
- **Provide Context and Intent**: Specify the purpose (e.g., "Create a logo for a high-end, minimalist skincare brand").
- **Iterate and Refine**: Use follow-up prompts to adjust results (e.g., "Make the lighting warmer").
- **Use Step-by-Step Instructions**: For complex scenes, break the prompt into steps (e.g., "First, create a misty forest background...").
- **Use Semantic Negative Prompts**: Describe desired scenes positively (e.g., "an empty street" instead of "no cars").
- **Control the Camera**: Use terms like "wide-angle shot" or "low-angle perspective" for composition.

**Limitations**:
- Best performance in languages: EN, es-MX, ja-JP, zh-CN, hi-IN.
- Does not support audio or video inputs.
- Model may not return the exact number of requested images.
- Works best with 2-3 input images.
- For text in images, generate text first, then request the image.
- Images of children are not supported in EEA, CH, and UK.
- All generated images include a SynthID watermark.
"""


@app.post(
    "/generate",
    tags=["Image Generation"],
    response_model=GenerateResponse,
    description=API_DESCRIPTION,
)
# --- Change to async def ---
async def generate_image(
    images: List[UploadFile] = Form(...),  # Use Form for multipart data
    prompt: str = Form(...),
):
    try:
        # Validate number of images (between 2 and 3)
        if len(images) < 2 or len(images) > 3:
            raise HTTPException(
                status_code=400,
                detail="Exactly 2 or 3 images are required (e.g., dress and model, or dress, model, and background)",
            )

        # Validate prompt using Pydantic method
        try:
            # Call the class method correctly
            cleaned_prompt = GenerateRequest.prompt_must_be_string(prompt)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Read and validate uploaded images
        image_parts = []
        filenames = []
        for i, image in enumerate(images):
            image_bytes = await image.read()  # This is correct for async FastAPI
            # Use the actual content type from the UploadFile if available
            mime_type = (
                image.content_type
                or mimetypes.guess_type(image.filename)[0]
                or (
                    "image/jpeg"
                    if image.filename
                    and image.filename.lower().endswith((".jpg", ".jpeg"))
                    else "image/png"
                )
            )
            if not mime_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image file type for {image.filename}",
                )
            # Create Part from bytes using the async client's types module
            image_parts.append(
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            )
            filenames.append(image.filename)

        # print image parts for debugging
        print(image_parts)
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

        # Debug: Print contents structure
        print("Contents prepared for generation:", contents)

        # Generate content config (dictionary-based is fine)
        generate_content_config = {
            "response_modalities": ["TEXT", "IMAGE"],
            "max_output_tokens": 200,
        }

        # --- Use await with the asynchronous client ---
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        # Debug: Print response structure
        print("Generation response received:", response)

        # Extract text output (if any)
        text_output = response.text if response.text else None

        # Extract image parts - Assuming inline_data is base64 encoded bytes
        # Check the SDK docs or response structure if this doesn't work
        image_parts_data = [
            part.inline_data.data  # This should be the base64 string
            for candidate in response.candidates
            for part in candidate.content.parts
            if part.inline_data
            and part.inline_data.data
            and part.inline_data.mime_type
            and part.inline_data.mime_type.startswith("image/")
        ]

        if not image_parts_data:
            raise HTTPException(
                status_code=500, detail="No image generated in response"
            )

        # Save the first generated image (assuming one for simplicity)
        # Decode the base64 data
        try:
            image_data_bytes = base64.b64decode(image_parts_data[0])
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to decode generated image data: {e}"
            )

        generated_filename = f"generated_{uuid.uuid4().hex}.png"  # Assuming PNG, adjust if needed based on mime_type
        generated_path = os.path.join("./images", generated_filename)

        # Write bytes directly to file
        try:
            with open(generated_path, "wb") as f:
                f.write(image_data_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save generated image: {e}"
            )

        # Store in DB (store all input filenames as a comma-separated string)
        generation_id = uuid.uuid4().hex
        timestamp = datetime.now().isoformat()
        input_filenames = ",".join(filenames)
        # Synchronous DB call within async function - acceptable for quick ops
        cursor.execute(
            """
            INSERT INTO generations (id, prompt, input_filenames, generated_filename, timestamp, text_output)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                generation_id,
                cleaned_prompt,
                input_filenames,
                generated_filename,
                timestamp,
                text_output,
            ),
        )
        conn.commit()

        # Return clickable URL (adjust host/port if needed for your setup)
        image_url = f"http://localhost:8000/images/{generated_filename}"

        return GenerateResponse(
            image_url=image_url, generation_id=generation_id, text_output=text_output
        )

    except genai.errors.APIError as e:  # Ensure correct error import path
        raise HTTPException(
            status_code=500, detail=f"GenAI API Error: {e.code} - {e.message}"
        )
    except HTTPException:  # Re-raise HTTPExceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# Run the app (for development: uvicorn main:app --reload)
# This part is usually handled by the uvicorn command line
# if __name__ == "__main__":
#     import uvicorn
#     # reload=True is passed via command line: uvicorn main:app --reload
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
