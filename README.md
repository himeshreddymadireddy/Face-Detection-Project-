# Face Recognition Attendance System

A Flask-based web application for detecting and recognizing student faces in uploaded images. Matches detected faces against a pre-loaded database of student face encodings and provides visual results.

## Features

- **Face Detection**: Detects faces in uploaded images using `face_recognition` library.
- **Student Matching**: Compares detected faces with pre-loaded student encodings.
- **Visual Output**: Draws bounding boxes around faces with labels (matched student ID or "Unknown").
- **Results Summary**: Displays lists of detected and undetected students.
- **Simple Web Interface**: Built with Flask for easy image upload and result display.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/himeshreddymadireddy/Face-Detection-Project-
   cd Face-Detection-Project-
2. **Install dependencies**:

```bash
pip install flask numpy opencv-python face-recognition dlib
```
- **Note**: On some systems, dlib may require additional steps. See dlib installation guide.

3. **Folder Structure**:
Ensure the following folders exist (created automatically on first run):

```bash
- uploads/       # Stores uploaded images
- outputs/       # Stores processed images with bounding boxes
- student_faces/ # Contains subfolders (e.g., "student_01", "student_02") with student images
```
## Configuration
1. **Add Student Faces**:

- FOlDER CREATION:Create a folder for each student in student_faces/ (e.g., student_faces/student_01).
- Place 1+ clear face images (JPG/PNG) in each student's folder. The app will auto-generate encodings on startup.

2. **Pre-Trained Models**:

* The app uses two pre-trained models:
  * shape_predictor_58_free_landmarks.dat: For face landmark detection.
  * dis_lace_recognition_resnet_model_v1.dat: For face encoding.
* Ensure these files are present in the root directory.

## Usage
1. Run the app:
```bash
python app.py
```
2. Access the web interface:
  Open http://localhost:5000 in a browser.
3. Upload an image:

* Use the web form to upload an image containing faces.
* Processed results will show:

  * Detected students (green bounding boxes).

  * Undetected students (red bounding boxes labeled "Unknown").

  * Lists of recognized and unrecognized student IDs.

## API Endpoints
* GET /: Homepage with upload form.
* POST /upload: Handle image upload and processing.
* GET /outputs/<filename>: Retrieve processed images.

## Dependencies
* Flask (web framework)

* face-recognition (face detection/encoding)

* OpenCV (image processing)

* dlib (machine learning backend)

* numpy (array handling)

## Acknowledgments
* Face recognition powered by face-recognition library.
* Pre-trained models from dlib.

## Notes
- **Tolerance Setting**: Matches are determined with a tolerance of 0.6 (lower = stricter). Adjust in detect_and_match_faces() if needed.

- **Image Quality**: For best results, use high-quality student images with clear frontal faces.
