import cv2

# loading pre-trained front-facing faces data
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# to capture video from webcam
webcam = cv2.VideoCapture(0)

print("\n\n\033[1;32m PRESS Q TO QUIT PROGRAM\033[0m   \n\n")

# loops through frames
while True:

    # reads current frame
    successful_frame_read, frame = webcam.read()

    # converts image to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects faces 
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draws rectangle around face(s) in a frame
    for i in range(len(face_coordinates)):
        (x, y, w, h) = face_coordinates[i]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (79, 252, 5), 2)

    # shows each frame
    cv2.imshow("People", frame)

    # waits for key press   
    key = cv2.waitKey(1)

    # stops loop if Q (ASCII form) is pressed
    if key == 81 or key == 113:
        break

# stops webcam
webcam.release()





    

