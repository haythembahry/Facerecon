# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.



def app():
    st.title("Welcome to Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    # Sidebar content
    st.sidebar.write("## Instructions:")
    st.sidebar.write("1. Click on 'Detect Faces' to open the web camra")
    st.sidebar.write("2. click on 'Save image' to save an image")
    color = st.color_picker("Pick a color", "#FF0000")
    # Create a slider to adjust the minNeighbors parameter
    min_neighbors_range = (1, 5)  # Adjust the range based on your needs
    min_neighbors = st.slider("Select minNeighbors", min_neighbors_range[0], min_neighbors_range[1])
    # Create a slider to adjust the scaleFactor parameter
    scale_factor_range = (1.1, 1.5)  # Adjust the range based on your needs
    scale_factor = st.slider("Select scaleFactor", scale_factor_range[0], scale_factor_range[1])

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function
        detect_faces()

    # Create a video capture object
    cap = cv2.VideoCapture(0)
    # Create a button to capture a picture
    if st.button('Capture image'):
        # Read a frame from the camera
        ret, frame = cap.read()

        if ret:
            # Display the captured frame
            st.image(frame, channels='BGR')

            # Save the captured frame as an image
            cv2.imwrite('captured_image.jpg', frame)

            # Show a success message
            st.write('Image captured successfully.')

if __name__ == "__main__":
    import cv2
    import streamlit as st

    face_cascade = cv2.CascadeClassifier('C:/Users/x/haarcascade_frontalface_default.xml')
    app()
    # Add a color picker widget