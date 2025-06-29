import cv2
import numpy as np
import pyvirtualcam
import mediapipe as mp

# Load and resize your background image
background_img = cv2.imread("F:\\Abeli\\Downloads\\360_F_712771267_L4Vx8541vakOmUi6rpCRvHlvVvgEEX5e.jpg")  # Replace with your image path

mp_selfie_segmentation = mp.solutions.selfie_segmentation
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)   # Range typically [0.0, 1.0] or [0, 255]
# cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
# cap.set(cv2.CAP_PROP_SATURATION, 0.6)
# cap.set(cv2.CAP_PROP_EXPOSURE, -4)      # Use trial-and-error, values vary by camera
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation, \
     pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        bg_resized = cv2.resize(background_img, (640, 480))
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb)

        # Use float mask for smooth blending
        mask = results.segmentation_mask
        mask = cv2.resize(mask, (640, 480))[:, :, np.newaxis]  # Ensure correct shape

        # Normalize to range [0, 1]
        mask = np.clip(mask, 0, 1)

        # Blend using weighted sum
        output = (frame * mask + bg_resized * (1 - mask)).astype(np.uint8)

        # Send to virtual cam
        cam.send(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()

        # Optional preview
        cv2.imshow("Virtual Cam Output", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
