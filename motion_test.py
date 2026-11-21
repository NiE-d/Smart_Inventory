
import cv2
import numpy as np
import urllib.request
import time

IMAGE_URL = "http://127.0.0.1:5000/image"

# Estimate the stationary background, then detect moving objects.
background = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=35,
    detectShadows=False
)

previous_x = None
last_seen = 0
frame_count = 0
direction = "Waiting for movement"

print("Smart Inventory - Motion Test")
print("Keep the camera still and use a plain background.")
print("Move a product horizontally through the view.")
print("Press Q in the video window to quit.")

while True:
    try:
        # Request the latest image without browser-style caching.
        url = f"{IMAGE_URL}?t={time.time_ns()}"

        with urllib.request.urlopen(url, timeout=5) as response:
            data = response.read()

        frame = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            continue

        frame_count += 1
        height, width = frame.shape[:2]

        # Learn the stationary scene during the initial frames.
        mask = background.apply(frame)

        # Remove small noise.
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Ignore small moving regions.
        candidates = [
            c for c in contours if cv2.contourArea(c) > 500
        ]

        if frame_count < 25:
            direction = "Learning background"

        elif candidates:
            # Track the largest foreground region.
            contour = max(candidates, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )
            cv2.circle(
                frame, (center_x, y + h // 2),
                4, (0, 0, 255), -1
            )

            if previous_x is not None:
                movement = center_x - previous_x

                if movement > 5:
                    direction = "Moving RIGHT"
                elif movement < -5:
                    direction = "Moving LEFT"

            previous_x = center_x
            last_seen = time.time()

        # Reset stale tracking after the object disappears.
        if time.time() - last_seen > 1.0:
            previous_x = None
            if frame_count >= 25:
                direction = "Waiting for movement"

        cv2.putText(
            frame,
            direction,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.imshow("Smart Inventory - Motion Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as error:
        print("Waiting for image:", error)
        time.sleep(0.2)

cv2.destroyAllWindows()
