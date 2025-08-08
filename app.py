from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from datetime import datetime
import os
import base64
import io
from PIL import Image
import numpy as np
import hashlib

app = Flask(__name__)

# Directory containing known face images
known_faces_dir = 'images'

# Store known face features for lightweight comparison
known_face_features = {}
known_names = []

# Load known faces using PIL with feature extraction
def load_known_faces():
    global known_face_features, known_names
    known_face_features = {}
    known_names = []
    
    # Load actual face images from the images directory
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(known_faces_dir, filename)
            try:
                image = Image.open(image_path)
                # Convert to grayscale and resize for consistent feature extraction
                image = image.convert('L').resize((100, 100))
                
                # Extract features using a simple but effective method
                features = extract_features(image)
                
                known_face_features[filename.split('.')[0]] = features
                known_names.append(filename.split('.')[0])
                print(f"Loaded face: {filename.split('.')[0]}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(known_names)} known faces: {known_names}")

def extract_features(image):
    """Extract features from image using a lightweight approach"""
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Apply simple blur using convolution
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    img_blur = np.zeros_like(img_array)
    
    # Simple convolution for blurring
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            img_blur[i, j] = np.sum(img_array[i-1:i+2, j-1:j+2] * kernel)
    
    # Extract features using gradient and intensity information
    features = []
    
    # Divide image into 10x10 blocks and extract features from each
    block_size = 10
    for i in range(0, 100, block_size):
        for j in range(0, 100, block_size):
            block = img_blur[i:i+block_size, j:j+block_size]
            
            # Calculate mean intensity
            mean_intensity = np.mean(block)
            
            # Calculate gradient magnitude (simplified and fixed)
            if block.shape[0] > 1 and block.shape[1] > 1:
                grad_x = np.diff(block, axis=1)
                grad_y = np.diff(block, axis=0)
                # Ensure both gradients have the same shape
                min_rows = min(grad_x.shape[0], grad_y.shape[0])
                min_cols = min(grad_x.shape[1], grad_y.shape[1])
                if min_rows > 0 and min_cols > 0:
                    gradient_magnitude = np.sqrt(grad_x[:min_rows, :min_cols]**2 + grad_y[:min_rows, :min_cols]**2)
                    mean_gradient = np.mean(gradient_magnitude)
                else:
                    mean_gradient = 0
            else:
                mean_gradient = 0
            
            # Calculate standard deviation
            std_intensity = np.std(block)
            
            # Calculate histogram features
            hist, _ = np.histogram(block, bins=8, range=(0, 255))
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize with small epsilon
            
            features.extend([mean_intensity, mean_gradient, std_intensity])
            features.extend(hist)
    
    # Normalize features
    features = np.array(features)
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    return features

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-8)

# Function to recognize face using feature comparison
def recognize_face(image_data):
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize for consistent feature extraction
        image = image.convert('L').resize((100, 100))
        
        # Extract features from the uploaded image
        uploaded_features = extract_features(image)
        
        # Compare with known faces
        best_match = None
        best_similarity = 0
        
        for name, known_features in known_face_features.items():
            # Calculate cosine similarity (higher is better)
            similarity = cosine_similarity(uploaded_features, known_features)
            
            if similarity > best_similarity and similarity > 0.6:  # 60% similarity threshold
                best_similarity = similarity
                best_match = name
        
        if best_match:
            return best_match, f"Recognized as {best_match} (similarity: {best_similarity:.2f})"
        else:
            return None, "Face not recognized. Please try again."
            
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Function to mark attendance
def mark_attendance(student_name, file='attendance.xlsx'):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    try:
        # For Vercel deployment, we'll use a simple in-memory storage
        # In production, you'd use a database
        attendance_data = getattr(app, 'attendance_data', [])
        
        # Check if attendance already marked for today
        today_attendance = [record for record in attendance_data 
                          if record['Name'] == student_name and record['Date'] == current_date]
        
        if len(today_attendance) > 0:
            return False, f"Attendance already marked for {student_name} today."
        
        new_record = {"Name": student_name, "Date": current_date, "Time": current_time}
        attendance_data.append(new_record)
        app.attendance_data = attendance_data
        
        return True, f"Attendance marked for {student_name} at {current_time}"
        
    except Exception as e:
        return False, f"Error marking attendance: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received'})
        
        student_name, message = recognize_face(image_data)
        
        if student_name:
            success, attendance_message = mark_attendance(student_name)
            return jsonify({
                'success': True,
                'student_name': student_name,
                'message': message,
                'attendance_marked': success,
                'attendance_message': attendance_message
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/attendance')
def get_attendance():
    try:
        # Return in-memory attendance data
        attendance_data = getattr(app, 'attendance_data', [])
        return jsonify({'success': True, 'data': attendance_data})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/known-faces')
def get_known_faces():
    return jsonify({'success': True, 'faces': known_names})

# Initialize the app
if __name__ == '__main__':
    load_known_faces()
    print(f"Loaded {len(known_names)} known faces: {known_names}")
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For Vercel deployment
    load_known_faces()
