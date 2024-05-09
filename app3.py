import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Initialize cascade classifiers
face_cascade = cv2.CascadeClassifier("D:/kishan/Git_and_github/face-detect/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("D:/kishan/Git_and_github/face-detect/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("D:/kishan/Git_and_github/face-detect/haarcascade_smile.xml")

# Function to start video streaming
def start_video_stream():
    global video_stream
    video_stream = cv2.VideoCapture(0)
    show_frame()

# Function to stop video streaming
def stop_video_stream():
    if 'video_stream' in globals():
        video_stream.release()
        cv2.destroyAllWindows()

# Function to show frame in Tkinter window
def show_frame():
    _, frame = video_stream.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minSize=(10, 10))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 0, 255), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minSize=(20, 20))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (255, 0, 0), 2)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)
    label.img = img
    label.config(image=img)
    label.after(10, show_frame)

# Create main Tkinter window
root = tk.Tk()
root.title("Face Detection")

# Create a label to display the video stream
label = tk.Label(root)
label.pack()

# Create start and stop buttons
start_button = tk.Button(root, text="Start", command=start_video_stream)
start_button.pack(side=tk.LEFT)
stop_button = tk.Button(root, text="Stop", command=stop_video_stream)
stop_button.pack(side=tk.LEFT)

# Start the Tkinter event loop
root.mainloop()
