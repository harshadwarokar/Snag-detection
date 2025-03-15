# Snag Detection  

## Overview  
Snag Detection is a **Streamlit-based application** that leverages **Google Generative AI** to analyze images and videos of newly constructed or renovated rooms. The tool identifies construction defects such as **cracks, leaks, wire damage, electrical board faults, incomplete finishes, and other structural issues**.  

## Features  
- **Upload Images and Videos**: Supports `.jpg`, `.jpeg`, `.png`, `.mp4`, `.avi`, and `.mov` formats.  
- **AI-Powered Defect Detection**: Uses Google Gemini AI to analyze frames and provide defect descriptions.  
- **Frame Extraction from Videos**: Captures 1 frame per second for analysis.  
- **User-Friendly UI**: Built with Streamlit for easy interaction.  

## Installation  

### Prerequisites  
Ensure you have **Python 3.8+** installed on your system.  

### Setup  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/harshadwarokar/snag-detection.git
   cd snag-detection
   ```

2. **Create a virtual environment (Recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**  
   - Create a `.env` file in the root directory and add your **Google Generative AI API key**:  
     ```plaintext
     api_key = "YOUR_GOOGLE_GENERATIVE_AI_KEY"
     ```

## Usage  

Run the Streamlit app:  
```bash
streamlit run appstreamlit.py
```

### Uploading Files  
- **Images**: Upload an image, and the AI will detect defects.  
- **Videos**: Upload a video, and the app extracts frames to detect issues.  

## Dependencies  
- **Streamlit** (UI framework)  
- **OpenCV** (Image processing)  
- **Google Generative AI** (Defect detection)  
- **Pillow** (Image handling)  
- **NumPy** (Array manipulation)  

## Contributing  
Pull requests are welcome! If you have suggestions or improvements, please open an issue or submit a PR.  

