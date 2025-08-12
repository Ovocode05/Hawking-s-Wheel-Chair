from turtle import distance
import cv2
import numpy as np

# -----------------------------------
# Utility functions
# -----------------------------------

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv

def get_mask(hsv, lower, upper, use_morph=True):
    mask = cv2.inRange(hsv, lower, upper)
    if use_morph:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def get_centers(mask, min_area=100):
    """Return all contour centers above area threshold, sorted top to bottom (by y)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    # Sort top to bottom (forehead should be higher than chin)
    centers.sort(key=lambda pt: pt[1])
    return centers

def get_center(mask, min_area=100):
    centers = get_centers(mask, min_area)
    return centers[0] if centers else None

def calculate_angle(a, b, c):
    """Angle ABC in degrees"""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_annotations(image, forehead, joint, chin,outside, angle_deg1, angle_deg2, distance=None):
    cv2.circle(image, forehead, 5, (255, 0, 0), -1)  # Blue forehead
    cv2.circle(image, joint, 5, (0, 255, 0), -1)     # Green joint
    cv2.circle(image, chin, 5, (255, 0, 0), -1)      # Blue chin
    cv2.line(image, forehead, joint, (0, 0, 0), 2)
    cv2.line(image, joint, chin, (0, 0, 0), 2)
    cv2.line(image, joint, outside, (0, 0, 0), 2)
    if distance is not None:
        cv2.circle(image, joint, distance, (0, 0, 0), 1)
    cv2.putText(image, f"Angle1: {np.abs(angle_deg1 - angle_deg2):.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# -----------------------------------
# Main script
# -----------------------------------

image = cv2.imread("img2.jpg")
if image is None:
    print("Error: Image not found.")
    exit()

hsv = preprocess_image(image)

# HSV Ranges
lower_green = (40, 50, 50)
upper_green = (80, 255, 255)

lower_blue = (90, 50, 50)
upper_blue = (130, 255, 255)

# Masks
green_mask = get_mask(hsv, lower_green, upper_green)
blue_mask = get_mask(hsv, lower_blue, upper_blue)

# Detection
green_pos = get_center(green_mask)
blue_centers = get_centers(blue_mask, min_area=50)
print(green_pos, blue_centers)
distance = int(np.linalg.norm(np.array(green_pos) - np.array(blue_centers[2]))) if green_pos and len(blue_centers) >= 2 else None
print(f"Distance: {distance} pixels" if distance is not None else "Distance not calculated")

if green_pos and len(blue_centers) >= 2:
    # Sort: [forehead_blue, chin_blue]
    blue_forehead, blue_chin, outside = sorted(blue_centers, key=lambda pt: pt[1])
    angle_deg1 = calculate_angle(blue_forehead, green_pos, blue_chin)
    angle_deg2 = calculate_angle(blue_forehead, green_pos, outside)
    draw_annotations(image, blue_forehead, green_pos, blue_chin,outside, angle_deg1, angle_deg2, distance)
else:
    print("One or more markers not detected.")
    if not green_pos: print("⚠️ Green (jaw joint) not detected")
    if len(blue_centers) < 2: print(f"⚠️ Only {len(blue_centers)} blue marker(s) detected")

cv2.imshow("Markers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()