import cv2
import numpy as np
import pandas as pd
# from sympy import true

# ----------------------------
# Helper Functions
# ----------------------------
def get_centers(mask, min_area=100):
    """Return all blob centers above min_area size."""
    centers = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers

def calculate_angle(a, b, c):
    """Angle at point b given points a, b, c."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)
  

# ----------------------------
# Video Processing
# ----------------------------
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define new dimensions
    new_width = 1250
    new_height = 700
    new_dimensions = (new_width, new_height)

    # Resize the image
    resized_image = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Blue markers (forehead & chin)
    lower_blue = (90, 50, 50)
    upper_blue = (130, 255, 255)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Green marker (jaw joint)
    lower_green = (40, 50, 50)
    upper_green = (80, 255, 255)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Get marker positions
    blue_points = get_centers(blue_mask, min_area=50)
    green_points = get_centers(green_mask, min_area=50)

    if len(blue_points) >= 2 and len(green_points) >= 1:
        # Sort blue points by Y position (top = forehead, bottom = chin)
        blue_points_sorted = sorted(blue_points, key=lambda p: p[1])
        forehead_pos = blue_points_sorted[0]
        chin_pos = blue_points_sorted[-1]
        green_pos = green_points[0]  # Assuming only one green marker

        # Calculate angle
        angle = calculate_angle(forehead_pos, green_pos, chin_pos)

        # Draw markers
        cv2.circle(resized_image, forehead_pos, 5, (255, 0, 0), -1)  # Blue
        cv2.circle(resized_image, chin_pos, 5, (255, 0, 0), -1)      # Blue
        cv2.circle(resized_image, green_pos, 5, (0, 255, 0), -1)     # Green

        # Draw lines
        cv2.line(resized_image, forehead_pos, green_pos, (0, 0, 0), 2)
        cv2.line(resized_image, green_pos, chin_pos, (0, 0, 0), 2)

        # Display angle
        cv2.putText(resized_image, f"Angle: {angle:.2f}°", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        cv2.putText(resized_image, "Markers missing", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    

    cv2.imshow("Jaw Tracking", resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 