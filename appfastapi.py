from fastapi import FastAPI, Query, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel
import cv2
import google.generativeai as genai
from PIL import Image
import os
import tempfile
import numpy as np
import time
import urllib.request
import urllib.parse

# Load environment variables
load_dotenv()

# Environment variables
API_KEY = os.getenv("api_key")

# Validate required environment variables
if not API_KEY:
    raise RuntimeError("Required environment variables are missing. Check .env configuration.")

# Configure Google Generative AI
genai.configure(api_key=API_KEY)

router = APIRouter()

# Function to process image and generate response
def get_gemini_response_for_frame(input_prompt, frame):
    model = genai.GenerativeModel('gemini-1.5-flash')
    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Generate content using image and prompt
    response = model.generate_content([input_prompt, image])
    return response.text

# Function to extract frames from video at 1-second intervals
def extract_frames_from_video(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Set frame_interval to 1 second

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

@router.get("/analyze_video")
async def analyze_video(file_path: str):
    # Encode URL to handle spaces and special characters
    file_path = urllib.parse.quote(file_path, safe=':/')
    
    # Check if the file_path is a URL (http or https)
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Handle image or video URL
        try:
            input_prompt = "Analyze the uploaded image or video for construction and functional defects. Identify issues such as cracks, water and other leaks, wire damage, electrical board faults, incomplete finishes, and window and door and functional defects and also other defects that should be not there. Focus strictly on detected defects and list them in one line using commas. Provide a really short, industry-level description of each issue with its title in bullet points."
            
            if file_path.endswith((".jpg", ".jpeg", ".png")):
                # If it's an image URL
                image = Image.open(urllib.request.urlopen(file_path))
                frame = np.array(image)
                result = get_gemini_response_for_frame(input_prompt, frame)
                return JSONResponse(content={"description": result})
            elif file_path.endswith((".mp4", ".avi", ".mov")):
                # If it's a video URL
                frames = extract_frames_from_video(file_path)
                frames_to_analyze = frames[:5]  # Limit to first 5 frames for analysis
                combined_description = []
                
                for frame in frames_to_analyze:
                    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                        result = get_gemini_response_for_frame(input_prompt, frame)
                        combined_description.append(result)
                        time.sleep(1)  # Avoid hitting API limits

                overall_description = " ".join(combined_description)
                return JSONResponse(content={"description": overall_description})
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
    else:
        return JSONResponse(content={"error": "Invalid file path. Please provide a valid image or video URL."}, status_code=400)

app = FastAPI()

app.include_router(router)
