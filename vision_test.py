
import cv2
import urllib.request
import numpy as np

IMAGE_URL = "http://127.0.0.1:5000/image"

print("Smart Inventory - Vision Test")
print("Waiting for camera images...")
print("Press Q to close the window.")

while True:
    try:
        # Get the latest image from your Flask server
        with urllib.request.urlopen(IMAGE_URL, timeout=5) as response:
            image_bytes = response.read()

        # Convert JPEG bytes into an OpenCV image
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if frame is None:
            print("Could not decode image.")
            continue

        # Display the image
        cv2.putText(
            frame,
            "Smart Inventory Camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Smart Inventory - Camera Test", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as error:
        print("Waiting for camera/server:", error)
        cv2.waitKey(500)

cv2.destroyAllWindows()
