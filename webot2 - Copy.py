# ===============================
# 1. IMPORT LIBRARIES
# ===============================

print("[INIT] Importing libraries...")

from controller import Robot, Camera, Speaker
import numpy as np
import cv2
import easyocr

print("[INIT] Libraries loaded successfully")


# ===============================
# 2. INITIALIZE ROBOT
# ===============================

print("[INIT] Initializing robot...")

robot = Robot()
timestep = int(robot.getBasicTimeStep())

print(f"[INIT] Robot initialized | Timestep: {timestep}")


# ===============================
# 3. INITIALIZE DEVICES
# ===============================

print("[INIT] Setting up devices...")

camera = robot.getDevice('camera')
camera.enable(timestep)
print("[INIT] Camera enabled")

speaker = robot.getDevice('speaker')
print("[INIT] Speaker ready")


# ===============================
# 4. INITIALIZE OCR READER
# ===============================

print("[INIT] Loading OCR model (this may take a few seconds)...")

reader = easyocr.Reader(['en'], gpu=True)

print("[INIT] OCR model loaded successfully")


# ===============================
# 5. CAMERA IMAGE CAPTURE FUNCTION
# ===============================

def capture_image():
    print("[CAMERA] Capturing image...")

    width = camera.getWidth()
    height = camera.getHeight()

    image = camera.getImage()

    if image is None:
        print("[CAMERA] WARNING: No image received!")
        return None

    img = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    print("[CAMERA] Image captured successfully")

    return img_bgr


# ===============================
# 6. SIGN DETECTION
# ===============================

def detect_sign(image):
    print("[VISION] Running sign detection (currently passthrough)")
    return image


# ===============================
# 7. OCR TEXT EXTRACTION
# ===============================

def extract_text(image):
    print("[OCR] Running text extraction...")

    results = reader.readtext(image)

    if not results:
        print("[OCR] No text detected")
        return ""

    # Extract only the detected text strings
    extracted_text = " ".join([res[1] for res in results]).strip()

    # Print raw OCR output (bounding boxes + confidence)
    print("[OCR] Raw result:", results)

    # ✅ Print final detected text (what your system will use)
    print("[OCR] Detected:", extracted_text)

    return extracted_text

# ===============================
# 8. SPEECH FUNCTION WITH STATUS
# ===============================

is_speaking = False
speech_end_time = 0

def speak_command(text):
    global is_speaking, speech_end_time

    if text == "":
        print("[TTS] No text to speak")
        return

    print("[TTS] Speaking:", text)

    speaker.speak(text, 1.0)

    duration = max(1.5, len(text) * 0.06)

    speech_end_time = robot.getTime() + duration
    is_speaking = True


# ===============================
# 9. MAIN LOOP
# ===============================

print("[SYSTEM] Starting main control loop...")

step_counter = 0

while robot.step(timestep) != -1:

    step_counter += 1

    # Heartbeat (every 50 steps)
    if step_counter % 50 == 0:
        print(f"[LOOP] Running... Step: {step_counter} | Time: {robot.getTime():.2f}s")

    # Check if speech finished
    if is_speaking and robot.getTime() >= speech_end_time:
        print("[TTS] Done")
        is_speaking = False

    # Skip processing if speaking
    if is_speaking:
        print("[LOOP] Waiting for speech to finish...")
        continue

    # Run OCR periodically
    if step_counter % 20 == 0:

        print("[PIPELINE] Starting perception pipeline...")

        # Capture image
        image = capture_image()

        if image is None:
            continue

        # Detect sign
        sign = detect_sign(image)

        # Extract text
        text = extract_text(sign)

        if text:
            print("[OCR] Detected:", text)
            speak_command(text)
        else:
            print("[PIPELINE] No readable text this cycle")