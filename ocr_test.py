
import cv2
import numpy as np
import pytesseract
import urllib.request
import time

IMAGE_URL = "http://127.0.0.1:5000/image"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

print("SMART INVENTORY - OCR TEST")
print("Wait for the image to appear.")
print("Select the printed label using your mouse.")
print("Press ENTER or SPACE to confirm selection.")
print("Press Q to exit.")

def get_latest_frame():
    url = f"{IMAGE_URL}?t={time.time_ns()}"
    with urllib.request.urlopen(url, timeout=5) as response:
        data = response.read()

    frame = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError("Invalid camera image")

    return cv2.flip(frame, 1)


def recognize_text(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Enlarge the selected label
    gray = cv2.resize(
        gray, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # Try multiple image-processing methods
    _, otsu = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

    candidates = [
        ("Grayscale", enhanced),
        ("Otsu", otsu),
        ("Adaptive", adaptive)
    ]

    best_text = ""
    best_score = -1
    best_image = enhanced

    for method_name, image in candidates:
        for psm in (6, 11, 12):
            config = f"--oem 3 --psm {psm}"

            result = pytesseract.image_to_data(
                image,
                lang="eng",
                config=config,
                output_type=pytesseract.Output.DICT
            )

            words = []
            confidences = []

            for word, confidence in zip(
                result["text"], result["conf"]
            ):
                word = word.strip()

                try:
                    confidence = float(confidence)
                except ValueError:
                    continue

                if word and confidence >= 0:
                    words.append(word)
                    confidences.append(confidence)

            if confidences:
                text = " ".join(words)
                # Prefer readable text with higher average confidence
                score = sum(confidences) / len(confidences)

                if score > best_score:
                    best_score = score
                    best_text = text
                    best_image = image

    return best_text, best_score, best_image


try:
    # Get an image to select the label region
    while True:
        frame = get_latest_frame()
        cv2.imshow("Select Product Label", frame)

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            raise SystemExit

        # Press S to select the label
        if key == ord("s"):
            break

    cv2.destroyWindow("Select Product Label")

    x, y, w, h = cv2.selectROI(
        "Select Printed Label",
        frame,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyWindow("Select Printed Label")

    if w == 0 or h == 0:
        print("No label selected. Restart and press S.")
        raise SystemExit

    last_text = None
    last_print_time = 0

    while True:
        frame = get_latest_frame()

        # Crop the same label area in each frame
        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        text, confidence, processed = recognize_text(crop)

        if text != last_text and (
            time.time() - last_print_time > 1
        ):
            print("\nRecognized text:", text or "No text")
            print("Average OCR confidence:", round(confidence, 1))

            last_text = text
            last_print_time = time.time()

        cv2.imshow("Mirrored Camera", frame)
        cv2.imshow("Selected Label", crop)
        cv2.imshow("Best OCR Preprocessing", processed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
