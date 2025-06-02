#!/usr/bin/env python3
"""
Simple test client for the Image Processing API.
This script demonstrates how to use the API to extract data from images.
"""

import requests
import json
import sys
import os

API_BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Make sure the server is running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def extract_image_data(image_path):
    """Extract data from an image file."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return False
    
    print(f"\nExtracting data from: {image_path}")
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(f"{API_BASE_URL}/extract", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! Extracted data:")
            
            # Print extracted text
            if data['extracted_text']['success']:
                print(f"\n📝 Extracted Text:")
                print(f"Raw Text: {data['extracted_text']['raw_text'][:200]}...")
                print(f"Number of text elements: {len(data['extracted_text']['detailed_text'])}")
            else:
                print(f"❌ Text extraction failed: {data['extracted_text'].get('error', 'Unknown error')}")
            
            # Print image metadata
            print(f"\n📊 Image Metadata:")
            metadata = data['metadata']
            print(f"Format: {metadata.get('format', 'Unknown')}")
            print(f"Size: {metadata['size']['width']}x{metadata['size']['height']}")
            print(f"Mode: {metadata.get('mode', 'Unknown')}")
            print(f"File Size: {data['file_size']} bytes")
            
            # Print color analysis
            if 'color_analysis' in data and 'dominant_colors' in data['color_analysis']:
                print(f"\n🎨 Color Analysis:")
                colors = data['color_analysis']['dominant_colors']
                for i, color in enumerate(colors[:3]):  # Show top 3 colors
                    print(f"Color {i+1}: {color['hex']} ({color['percentage']:.1f}%)")
            
            return True
        else:
            error_data = response.json()
            print(f"❌ Error: {error_data.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Make sure the server is running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function to run the test client."""
    print("🔍 Image Processing API Test Client")
    print("=" * 40)
    
    # Test health check
    if not test_health_check():
        print("\n❌ Health check failed. Please start the API server first.")
        return
    
    print("\n✅ API is running!")
    
    # Test image extraction
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        extract_image_data(image_path)
    else:
        print("\n📁 To test image extraction, provide an image path:")
        print(f"   python {sys.argv[0]} path/to/your/image.jpg")
        
        # Look for common image files in current directory
        current_dir_images = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'webp']:
            for file in os.listdir('.'):
                if file.lower().endswith(f'.{ext}'):
                    current_dir_images.append(file)
        
        if current_dir_images:
            print(f"\n📷 Found images in current directory: {', '.join(current_dir_images)}")
            test_image = current_dir_images[0]
            print(f"\n🧪 Testing with: {test_image}")
            extract_image_data(test_image)

if __name__ == "__main__":
    main() 