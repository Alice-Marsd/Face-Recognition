# Import OpenCV2 for image processing
import cv2,os

# Import numpy for matrices calculations
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
print('current_dir='+current_dir)

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read(current_dir+"//trainer/trainer.yml")

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id = recognizer.predict(gray[y:y+h,x:x+w])
      
        # Check the ID if exist 
        if(Id[0] == 1):
           user = "Jacky"
        # If not exist, then it is Unknown
        elif(Id[0] == 2):
            user = "Jenifer"
        elif(Id[0] == 3):
            user = "JiaTingTing"
        elif(Id[0] == 4):
            user = "TieHanHan"
        else:
            print(Id)
            user = "Unknow"

        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(user), (x,y-40), font, 2, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()