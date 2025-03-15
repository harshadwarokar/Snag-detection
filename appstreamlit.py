import streamlit as st
import cv2
import tempfile
import google.generativeai as genai
from PIL import Image
import numpy as np
import os
import time

# Configure Google Generative AI API key
API_KEY = "AIzaSyD_sZRCvGI3P7aPVyCtznPAuqtcBdiQYvo"
genai.configure(api_key=API_KEY)

# Function to extract frames from video at 1-second intervals
def extract_frames_from_video(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps)  # Extract 1 frame per second
    
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames

# Function to process a frame or image and generate a response
def analyze_frame(input_prompt, frame):
    model = genai.GenerativeModel('gemini-1.5-flash')
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    response = model.generate_content([input_prompt, image])
    generated_text = response.text if response else "No response received."
    
    detected_issues = []
    descriptions = []
    for sentence in generated_text.split(". "):
        if any(issue in sentence.lower() for issue in ["crack", "leak", "damage", "fault", "defect", "broken", "hazard"]):
            issue_type = sentence.split(":")[0]
            detected_issues.append(issue_type.strip())
            descriptions.append(sentence.strip())
    
    return detected_issues, "\n".join(descriptions) if detected_issues else ([], "No defects detected.")

# Streamlit App
st.title("Snag Detection for Newly Constructed or Renovated Rooms")
st.write("Upload an image or video to detect construction issues like cracks, leaks, electrical faults, and more.")

# User input prompt


input_prompt ="Analyze the uploaded image or video for . Identify issues such as cracks, water and other leaks, wire damage, electrical board faults, incomplete finishes, and window and door and functional defects and also other defects that should be not there . Focus strictly on detected defects and list them in one line  . Provide a really short, industry-level description of each issue with its title in bullet points ."

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                issues, descriptions = analyze_frame(input_prompt, frame)
                
                if issues:
                    st.write(f"**Detected Issues:** {', '.join(issues)}")
                    st.write(descriptions)
                else:
                    st.write("No significant issues detected.")
    
    elif file_type == "video":
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name

        st.video(uploaded_file)

        if st.button("Analyze Video"):
            with st.spinner("Extracting frames and analyzing..."):
                frames = extract_frames_from_video(temp_video_path)
                st.write(f"Extracted {len(frames)} frames for analysis.")
                all_issues = set()
                all_descriptions = []

                for frame in frames:
                    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                        issues, descriptions = analyze_frame(input_prompt, frame)
                        if issues:
                            all_issues.update(issues)
                            all_descriptions.append(descriptions)
                        time.sleep(1)
                
                if all_issues:
                    st.write(f"**Detected Issues:** {', '.join(all_issues)}")
                    st.write("\n".join(all_descriptions))
                else:
                    st.write("No significant issues detected.")
