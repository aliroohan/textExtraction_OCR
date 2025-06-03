from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import base64
from PIL import Image, ExifTags
import pytesseract
import cv2
import numpy as np
from datetime import datetime
import hashlib

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(image_path):
    """Extract text from image using OCR."""
    try:
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(Image.open(image_path))
        
        # Also get detailed data including confidence scores
        data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
        
        # Filter out empty text and low confidence results
        filtered_text = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30 and data['text'][i].strip():
                filtered_text.append({
                    'text': data['text'][i].strip(),
                    'confidence': int(data['conf'][i]),
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        return {
            'raw_text': text.strip(),
            'detailed_text': filtered_text,
            'success': True
        }
    except Exception as e:
        return {
            'raw_text': '',
            'detailed_text': [],
            'success': False,
            'error': str(e)
        }

def extract_image_metadata(image_path):
    """Extract metadata from image."""
    try:
        with Image.open(image_path) as img:
            # Basic image info
            metadata = {
                'format': img.format,
                'mode': img.mode,
                'size': {
                    'width': img.width,
                    'height': img.height
                },
                'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
            
            # EXIF data
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = str(value)
            
            metadata['exif'] = exif_data
            
            # File size
            metadata['file_size'] = os.path.getsize(image_path)
            
            return metadata
    except Exception as e:
        return {'error': str(e)}

    """Analyze dominant colors in the image."""
    try:
        # Load image with OpenCV
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape image to be a list of pixels
        pixels = img_rgb.reshape(-1, 3)
        
        # Calculate color statistics
        mean_color = np.mean(pixels, axis=0).astype(int).tolist()
        
        # Find dominant colors using k-means clustering
        from sklearn.cluster import KMeans
        
        # Use 5 clusters to find 5 dominant colors
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int).tolist()
        
        # Calculate color percentages
        labels = kmeans.labels_
        percentages = []
        total_pixels = len(labels)
        
        for i in range(5):
            percentage = (np.sum(labels == i) / total_pixels) * 100
            percentages.append(round(percentage, 2))
        
        # Combine colors with percentages
        dominant_colors = [
            {
                'color': {'r': color[0], 'g': color[1], 'b': color[2]},
                'hex': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                'percentage': percentages[i]
            }
            for i, color in enumerate(colors)
        ]
        
        # Sort by percentage
        dominant_colors.sort(key=lambda x: x['percentage'], reverse=True)
        
        return {
            'mean_color': {
                'r': mean_color[0], 
                'g': mean_color[1], 
                'b': mean_color[2]
            },
            'dominant_colors': dominant_colors
        }
    except Exception as e:
        return {'error': str(e)}


def draw_text_boxes(image_path, text_data):
    """Draw boxes around detected text regions."""
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        # Draw boxes for each detected text region
        for item in text_data['detailed_text']:
            bbox = item['bbox']
            # Draw rectangle
            cv2.rectangle(
                img,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                (0, 255, 0),  # Green color
                2  # Thickness
            )
        
        # Save the annotated image
        annotated_path = image_path.replace('.', '_annotated.')
        cv2.imwrite(annotated_path, img)
        return annotated_path
    except Exception as e:
        print(f"Error drawing text boxes: {str(e)}")
        return image_path

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint."""
    return jsonify({
        'message': 'Image Processing API is running',
        'version': '1.0.0',
        'endpoints': {
            'extract': '/extract - POST - Upload image for data extraction',
            'health': '/ - GET - Health check'
        }
    })

@app.route('/extract', methods=['POST'])
def extract_image_data():
    """Extract visual data from uploaded image."""
    
    # Check if image file is in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_hash = hashlib.md5(file.read()).hexdigest()[:8]
            file.seek(0)  # Reset file pointer
            
            filename = f"{timestamp}_{file_hash}_{file.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save uploaded file
            file.save(file_path)
            
            # Extract text using OCR
            text_data = extract_text_from_image(file_path)
            
            # Draw boxes around detected text
            annotated_image_path = draw_text_boxes(file_path, text_data)
            
            # Extract image metadata
            metadata = extract_image_metadata(file_path)
            
            # Convert annotated image to base64 for response
            with open(annotated_image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Clean up - remove uploaded files
            os.remove(file_path)
            if annotated_image_path != file_path:  # Only remove if it's a different file
                os.remove(annotated_image_path)
            
            # Prepare response
            response_data = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'original_filename': file.filename,
                'file_size': file_size,
                'extracted_text': text_data,
                'metadata': metadata,
                'image_base64': img_base64  # This now contains the annotated image
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up files if they exist
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            if 'annotated_image_path' in locals() and os.path.exists(annotated_image_path) and annotated_image_path != file_path:
                os.remove(annotated_image_path)
            
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 500
    
    else:
        return jsonify({
            'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
