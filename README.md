# AirWrite AI — Hand Gesture Writing & Character Recognition System

AirWrite AI is a real-time computer vision and deep learning project that allows users to write text in the air using hand gestures. The system recognizes handwritten strokes using a CNN trained on the EMNIST dataset and converts them into editable digital text.

---

## Features

- Real-time hand tracking using MediaPipe
- CNN-based character recognition (EMNIST byclass dataset)
- Air handwriting using index finger tracking
- Stroke-to-character conversion pipeline
- Cursor-based text insertion system
- Gesture-based editing (space, delete, newline)
- Smooth stroke tracking using temporal filtering (alpha smoothing)
- Real-time OpenCV rendering

---

## System Pipeline

### 1. Hand Tracking
MediaPipe detects 21 hand landmarks in real time using webcam input.

---

### 2. Stroke Capture
When drawing mode is active, index finger movement is recorded as a sequence of (x, y) coordinates.

---

### 3. Stroke Rendering
After the user stops drawing:
- The stroke is drawn on a blank canvas
- A binary image representation is created

---

### 4. Preprocessing Pipeline

The stroke is converted into a CNN-compatible image:

- Create empty black canvas
- Draw stroke as white pixels
- Apply Gaussian blur
- Flip image horizontally
- Extract bounding box of stroke
- Resize to 20×20
- Center into 28×28 canvas
- Normalize pixel values to [0, 1]

---

### 5. Character Recognition

A CNN trained on EMNIST (62 classes) predicts the character from the processed image.

---

### 6. Text Construction System

Predicted characters are inserted into a dynamic text buffer:
- Each character has (char, x, y)
- Cursor controls insertion position
- Characters shift automatically when inserting new ones

---

## Gesture Controls

### Gesture | Condition | Action 
- Index + Middle fingers up -> Draw stroke
- When stroke ends (`len(points) > 5`) -> Stroke is converted into a predicted character
- Thumb + Middle finger and sum(fingers) == 3 -> Action: Insert space
- Thumb + Ring finger (landmark 16) proximity and sum(fingers) == 3 -> Remove last character
- Fist (all fingers down) -> New line 

---

## Model Architecture

- Conv2D (32) + BatchNorm + MaxPool
- Conv2D (64) + BatchNorm + MaxPool
- Conv2D (128) + BatchNorm
- Flatten
- Dense(256) + Dropout(0.5)
- Dense(62 classes, Softmax)

## Dataset: 
EMNIST byclass 

## Classes: 
0–9, A–Z, a–z (62 total)

---

## Technologies Used

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- TensorFlow Datasets (EMNIST)
