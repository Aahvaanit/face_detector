import cv2

# loading pre-trained front-facing faces data
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

# to capture video from webcam
webcam = cv2.VideoCapture(0)

print("\n\n\033[1;32m PRESS Q TO QUIT PROGRAM \033[0m \n\n")

# loops through frames
while True:

    # reads current frame
    successful_frame_read, frame = webcam.read()

    # abort if error
    if not successful_frame_read:
        break

    # converts image to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects faces 
    faces = trained_face_data.detectMultiScale(grayscaled_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        
        # draws rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (79, 252, 5), 2)

        # creates the face sub-image
        face = frame[y:y+h, x:x+w]

        # changes face to grayscale
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20)

        # person is smiling if len(smile) > 0
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))

    # shows each frame
    cv2.imshow("SMILE DETECTOR", frame)

    # waits for key press   
    key = cv2.waitKey(1)

    # stops loop if Q (ASCII form) is pressed
    if key == 81 or key == 113:
        break

# stops webcam
webcam.release()
cv2.destroyAllWindows()