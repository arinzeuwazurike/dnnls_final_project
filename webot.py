import easyocr

# Initialize reader (downloads model automatically)
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if available

# Run OCR
results = reader.readtext('screenshot.png')

# Extract text only
extracted_text = " ".join([res[1] for res in results])

print("Extracted Text:\n", extracted_text)



print("Extracted Text:\n", extracted_text)
#%%
import asyncio
import edge_tts

text = extracted_text

async def speak():
    communicate = edge_tts.Communicate(text, voice="en-GB-LibbyNeural")
    await communicate.save("output.mp3")

from IPython.display import Audio
Audio("output.mp3")
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize devices
speaker = robot.getDevice('speaker')
camera = robot.getDevice('camera')

# Only Camera needs enabling; Speaker does NOT have an .enable() method
camera.enable(timestep)

# Configure Speaker
speaker.setLanguage('en-UK')

# Schedule the speech
speaker.speak("Hello, I am an e-puck robot.", 1.0)

# Main loop: The simulation must run for sound to play
while robot.step(timestep) != -1:
    # This loop allows Webots to process the 'speak' command
    pass

# 5. CAMERA INPUT (COMPUTER VISION)
# =========================================
# - Capture image from robot camera
# - Convert image to usable format
# - Display or store frames for processing

# =========================================
# 6. SIGN DETECTION (CV PREPROCESSING)
# =========================================
# - Image preprocessing (grayscale, thresholding)
# - Region of interest detection
# - Crop sign from image

# =========================================
# 7. OCR (TEXT EXTRACTION)
# =========================================
# - Apply easyocr
# - Extract text from cropped sign
# - Clean extracted text

# =========================================
# 8. NLP (COMMAND INTERPRETATION)
# =========================================
# - Map extracted text to commands:
#     TURN LEFT → LEFT
#     TURN RIGHT → RIGHT
#     DO NOT ENTER → BLOCK
# - Store navigation command

# =========================================
# 9. SPEECH SYNTHESIS
# =========================================
# - Convert command to spoken output
# - Example: "Turning left"
# - Use TTS library (e.g., pyttsx3 / edge-tts)
