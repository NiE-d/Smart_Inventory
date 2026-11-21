
from flask import Flask, request, send_file, make_response
from io import BytesIO
from datetime import datetime
import threading

app = Flask(__name__)

latest_image = None
latest_time = None
image_lock = threading.Lock()


@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Inventory</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background: #111;
                color: white;
                margin: 20px;
            }

            img {
                max-width: 95%;
                max-height: 75vh;
                object-fit: contain;
                border: 2px solid #555;
                border-radius: 8px;
            }

            #status {
                color: #aaa;
                margin: 12px;
            }
        </style>
    </head>
    <body>
        <h1>Smart Inventory</h1>
        <p>ESP32 Camera Receiver</p>
        <img id="camera" src="/image" alt="Waiting for camera image">
        <p id="status">Waiting for camera...</p>

        <script>
            const camera = document.getElementById("camera");
            const status = document.getElementById("status");

            function refreshImage() {
                const imageUrl = "/image?t=" + Date.now();

                const newImage = new Image();

                newImage.onload = function() {
                    camera.src = imageUrl;
                    status.textContent =
                        "Camera connected | Updated: " +
                        new Date().toLocaleTimeString();
                };

                newImage.onerror = function() {
                    status.textContent = "Waiting for camera image...";
                };

                newImage.src = imageUrl;
            }

            refreshImage();
            setInterval(refreshImage, 500);
        </script>
    </body>
    </html>
    """


@app.route("/upload", methods=["POST"])
def upload():
    global latest_image, latest_time

    image_data = request.get_data()

    if not image_data:
        return "No image received", 400

    with image_lock:
        latest_image = image_data
        latest_time = datetime.now().strftime("%H:%M:%S")

    print(f"[{latest_time}] Received {len(image_data)} bytes")
    return "Image received", 200


@app.route("/image")
def image():
    with image_lock:
        image_data = latest_image

    if image_data is None:
        return "No image received yet", 404

    response = make_response(
        send_file(
            BytesIO(image_data),
            mimetype="image/jpeg",
            max_age=0
        )
    )

    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"

    return response


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
