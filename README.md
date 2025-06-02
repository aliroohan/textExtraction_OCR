# Image Processing API

A Flask-based REST API that extracts visual data from images including text recognition (OCR), metadata extraction, and color analysis.

## Features

- **Text Extraction (OCR)**: Extract text from images using Tesseract OCR
- **Image Metadata**: Extract EXIF data, dimensions, format, and file information
- **Color Analysis**: Identify dominant colors and calculate color statistics
- **Multiple Format Support**: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
- **Base64 Encoding**: Optional return of processed image as base64

## Prerequisites

Before running this application, you need to install Tesseract OCR on your system:

### Windows
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install Tesseract (note the installation path)
3. Add Tesseract to your system PATH or set the path in code:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### macOS
```bash
brew install tesseract
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

## Installation

1. Clone or download this project
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

```bash
python main.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
- **URL**: `GET /`
- **Description**: Check if the API is running
- **Response**: JSON with API status and available endpoints

### 2. Extract Image Data
- **URL**: `POST /extract`
- **Description**: Upload an image and extract visual data
- **Request**: Multipart form data with `image` field
- **Response**: JSON with extracted data

## Usage Examples

### Using curl
```bash
# Upload an image for processing
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/extract
```

### Using Python requests
```python
import requests

url = "http://localhost:5000/extract"
files = {"image": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
data = response.json()
print(data)
```

### Using JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:5000/extract', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Response Format

The API returns a JSON response with the following structure:

```json
{
    "success": true,
    "timestamp": "2024-01-15T10:30:00.000Z",
    "original_filename": "document.jpg",
    "file_size": 1048576,
    "extracted_text": {
        "raw_text": "Extracted text from the image...",
        "detailed_text": [
            {
                "text": "Hello",
                "confidence": 95,
                "bbox": {"x": 100, "y": 50, "width": 80, "height": 25}
            }
        ],
        "success": true
    },
    "metadata": {
        "format": "JPEG",
        "mode": "RGB",
        "size": {"width": 1920, "height": 1080},
        "has_transparency": false,
        "exif": {...},
        "file_size": 1048576
    },
    "color_analysis": {
        "mean_color": {"r": 128, "g": 64, "b": 192},
        "dominant_colors": [
            {
                "color": {"r": 255, "g": 255, "b": 255},
                "hex": "#ffffff",
                "percentage": 45.6
            }
        ]
    },
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

## Configuration

You can modify these settings in `main.py`:

- `MAX_FILE_SIZE`: Maximum upload file size (default: 16MB)
- `ALLOWED_EXTENSIONS`: Supported file formats
- `UPLOAD_FOLDER`: Temporary upload directory
- OCR confidence threshold (default: 30)

## Error Handling

The API handles various error scenarios:

- **400**: Bad request (no file, invalid format, etc.)
- **413**: File too large
- **500**: Internal server error

## File Structure

```
.
├── main.py              # Main Flask application
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── uploads/            # Temporary upload directory (created automatically)
```

## Security Notes

- Files are temporarily stored and automatically deleted after processing
- CORS is enabled for cross-origin requests
- File size limits are enforced
- Only specific image formats are allowed

## Troubleshooting

1. **Tesseract not found**: Make sure Tesseract is installed and in your PATH
2. **Memory errors**: Reduce image size or increase system memory
3. **Permission errors**: Ensure the application has write permissions for the uploads folder

## License

This project is open source and available under the MIT License. 