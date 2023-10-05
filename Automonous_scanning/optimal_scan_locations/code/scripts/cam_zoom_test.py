import cv2

def set_zoom(webcam, zoom_value):
    # Check if the webcam supports zoom control
    if not webcam.set(cv2.CAP_PROP_ZOOM, zoom_value):
        print("Zoom control is not supported by the webcam.")
        return False
    
    return True

# Open the webcam
webcam = cv2.VideoCapture(2)

# Check if the webcam is opened successfully
if not webcam.isOpened():
    print("Failed to open the webcam.")
    exit()

# Capture an image without zoom
ret, frame = webcam.read()

# Check if the capture was successful
if not ret:
    print("Failed to capture image from the webcam.")
    exit()

# Display the original image
cv2.imshow("Original Image", frame)

# Set the desired zoom value (between 0.0 and 1.0)
zoom_value = 0.5

# Set the zoom value
if not set_zoom(webcam, zoom_value):
    exit()

# Capture an image with zoom
ret, zoomed_frame = webcam.read()

# Check if the capture was successful
if not ret:
    print("Failed to capture zoomed image from the webcam.")
    exit()

# Display the zoomed image
cv2.imshow("Zoomed Image", zoomed_frame)

# Wait for key press to exit
cv2.waitKey(0)

# Release the webcam and close any open windows
webcam.release()
cv2.destroyAllWindows()
