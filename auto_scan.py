import cv2
import numpy as np
import urllib.request
import time
from pathlib import Path
from collections import deque, Counter

import torch
import torch.nn as nn
from torchvision import transforms, models
from openpyxl import Workbook, load_workbook

# ---------- SETTINGS ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "component_classifier_v2.pth"
EXCEL_PATH = BASE_DIR / "inventory.xlsx"
IMAGE_URL = "http://127.0.0.1:5000/image"

MIN_OBJECT_AREA = 500
MIN_CONFIDENCE = 0.75       # Collect more useful predictions; consensus still protects Excel.
STABLE_SECONDS = 1.0
IMAGE_SIZE = 224
CLASSIFY_INTERVAL = 0.20    # At most 5 predictions/second.
MIN_PREDICTIONS = 3
MIN_LABEL_AGREEMENT = 0.60
MIN_DIRECTION_FRAMES = 4

# ---------- LOAD AI MODEL ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
class_names = checkpoint["class_names"]
image_size = checkpoint.get("image_size", IMAGE_SIZE)

model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- CREATE / LOAD EXCEL ----------
if EXCEL_PATH.exists():
    wb = load_workbook(EXCEL_PATH)
    ws = wb["Inventory"] if "Inventory" in wb.sheetnames else wb.active
else:
    wb = Workbook()
    ws = wb.active
    ws.title = "Inventory"
    ws.append(["Name", "Quantity"])

if ws.max_row == 1 and ws.cell(1, 1).value is None:
    ws.append(["Name", "Quantity"])
elif ws.cell(1, 1).value != "Name" or ws.cell(1, 2).value != "Quantity":
    ws.insert_rows(1)
    ws.cell(1, 1, "Name")
    ws.cell(1, 2, "Quantity")


def normalize_name(name):
    return str(name).strip().lower()


def find_row(name):
    target = normalize_name(name)
    for row in range(2, ws.max_row + 1):
        if normalize_name(ws.cell(row, 1).value or "") == target:
            return row
    return None


for name in class_names:
    if find_row(name) is None:
        ws.append([name, 0])
wb.save(EXCEL_PATH)

# ---------- IMAGE / CLASSIFICATION ----------
background = cv2.createBackgroundSubtractorMOG2(
    history=120, varThreshold=32, detectShadows=False
)
kernel = np.ones((5, 5), np.uint8)

scan_active = False
direction = None
last_motion_time = time.time()
predictions = deque(maxlen=5)
position_history = deque(maxlen=9)
previous_gray = None
flow_dx_total = 0.0
flow_samples = 0
frame_count = 0
last_classify_time = 0.0


def get_frame():
    url = f"{IMAGE_URL}?t={time.time_ns()}"
    with urllib.request.urlopen(url, timeout=3) as response:
        data = response.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return frame


def classify(image):
    if image is None or image.size == 0:
        return None, 0.0
    # OpenCV crop is BGR; convert to RGB before the model preprocessing.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    score, index = torch.max(probabilities, dim=0)
    return class_names[index.item()], float(score.item())


def update_inventory(name, delta):
    row = find_row(name)
    if row is None:
        ws.append([name, max(0, delta)])
        row = ws.max_row
    else:
        old_qty = ws.cell(row, 2).value or 0
        ws.cell(row, 2, max(0, int(old_qty) + delta))
    wb.save(EXCEL_PATH)
    print(f"Excel updated: {name} = {ws.cell(row, 2).value}")


print("SMART ELECTRONIC INVENTORY")
print("Model:", MODEL_PATH.name)
print("Excel:", EXCEL_PATH)
print("Swipe LEFT TO RIGHT to add; RIGHT TO LEFT to subtract.")
print("Show one component at a time. Press Q to exit.")
print(f"Device: {device}; classes: {class_names}")

try:
    while True:
        try:
            frame = get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1
            height, width = frame.shape[:2]
            now = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mask = background.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)

            median_flow_x = 0.0
            moving_pixel_count = 0
            if previous_gray is not None and frame_count >= 25:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                flow_x = flow[..., 0]
                flow_y = flow[..., 1]
                valid = (mask > 0) & (np.abs(flow_x) > 0.25) & (np.abs(flow_y) < 4.0)
                if np.count_nonzero(valid) >= 20:
                    median_flow_x = float(np.median(flow_x[valid]))
                    moving_pixel_count = int(np.count_nonzero(valid))
            previous_gray = gray

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            objects = [c for c in contours if cv2.contourArea(c) >= MIN_OBJECT_AREA]
            status = "Ready - show one component"
            box = None

            if frame_count < 25:
                status = "Learning background..."
            elif objects:
                # Track the largest motion contour, as in the existing working direction version.
                contour = max(objects, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                box = (x, y, w, h)
                position_history.append(x + w // 2)
                last_motion_time = now

                if not scan_active:
                    scan_active = True
                    direction = None
                    predictions.clear()
                    position_history.clear()
                    position_history.append(x + w // 2)
                    flow_dx_total = 0.0
                    flow_samples = 0
                    print("Swipe started; tracking optical flow...")

                # Lock direction from optical flow. Positive X = left-to-right.
                if direction is None and moving_pixel_count >= 20:
                    if abs(median_flow_x) >= 0.35:
                        flow_dx_total += median_flow_x
                        flow_samples += 1
                    if flow_samples >= MIN_DIRECTION_FRAMES:
                        threshold = max(2.5, width * 0.012)
                        if flow_dx_total >= threshold:
                            direction = "LEFT_TO_RIGHT"
                            print("Direction locked: LEFT_TO_RIGHT")
                        elif flow_dx_total <= -threshold:
                            direction = "RIGHT_TO_LEFT"
                            print("Direction locked: RIGHT_TO_LEFT")

                # Classify at a controlled interval. Add a little padding around the crop.
                if now - last_classify_time >= CLASSIFY_INTERVAL:
                    last_classify_time = now
                    margin = max(12, int(max(w, h) * 0.12))
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(width, x + w + margin)
                    y2 = min(height, y + h + margin)
                    crop = frame[y1:y2, x1:x2]
                    label, confidence = classify(crop)
                    if label and confidence >= MIN_CONFIDENCE:
                        predictions.append((label, confidence))
                        print(f"Prediction: {label} ({confidence:.0%}) | accepted samples: {len(predictions)}")

                if predictions:
                    counts = Counter(name for name, _ in predictions)
                    best_name, votes = counts.most_common(1)[0]
                    agreement = votes / len(predictions)
                    if votes >= MIN_PREDICTIONS and agreement >= MIN_LABEL_AGREEMENT:
                        display_name = f"{best_name} ({agreement:.0%} agreement)"
                    else:
                        display_name = f"Confirming: {best_name} ({votes}/{len(predictions)})"
                else:
                    display_name = "Identifying component..."

                direction_text = direction if direction else "Direction pending"
                status = f"{display_name} | {direction_text}"
            else:
                status = "Ready - show one component"

            # Finish swipe only after motion has been absent long enough.
            if scan_active and now - last_motion_time >= STABLE_SECONDS:
                if direction is None:
                    print("Swipe ignored: direction unclear; move farther across the frame.")
                elif len(predictions) < MIN_PREDICTIONS:
                    print(f"Swipe ignored: only {len(predictions)} confident predictions; need {MIN_PREDICTIONS}.")
                else:
                    counts = Counter(name for name, _ in predictions)
                    label, votes = counts.most_common(1)[0]
                    agreement = votes / len(predictions)
                    scores = [score for name, score in predictions if name == label]
                    avg_confidence = sum(scores) / len(scores) if scores else 0.0
                    print(f"Swipe result: {label}; agreement={agreement:.0%}; average confidence={avg_confidence:.0%}; samples={len(predictions)}")

                    if votes < MIN_PREDICTIONS or agreement < MIN_LABEL_AGREEMENT:
                        print("Inventory unchanged: not enough matching predictions.")
                    elif direction == "LEFT_TO_RIGHT":
                        update_inventory(label, +1)
                        print(f"ADDED: {label}")
                    elif direction == "RIGHT_TO_LEFT":
                        update_inventory(label, -1)
                        print(f"REMOVED: {label}")

                scan_active = False
                direction = None
                predictions.clear()
                position_history.clear()
                flow_dx_total = 0.0
                flow_samples = 0

            display = frame.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "SMART INVENTORY", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, status[:90], (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if scan_active:
                cv2.putText(display, f"Flow X: {median_flow_x:+.2f} | samples: {flow_samples}",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.imshow("Smart Inventory Scanner", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as error:
            print("Scanner error:", error)
            time.sleep(0.2)

finally:
    cv2.destroyAllWindows()
    try:
        wb.save(EXCEL_PATH)
        wb.close()
    except Exception as error:
        print("Excel save/close warning:", error)
    print("Smart Inventory scanner stopped.")
