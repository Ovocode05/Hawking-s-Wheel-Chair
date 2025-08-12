import cv2
import numpy as np
import pandas as pd

# ----------------------------
# Helper Functions
# ----------------------------
def get_centers(mask, min_area=100):
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
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# ----------------------------
# Video Processing
# ----------------------------
cap = cv2.VideoCapture("Silent Speech/Words/Bukhar/bukhar_anvut.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise ValueError("FPS is zero. Check video file.")
frame_duration = 1 / fps

results = []  # Store results for all frames

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    time_stamp = frame_idx * frame_duration

    # Resize for consistent processing
    resized = cv2.resize(frame, (1250, 700), interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Blue markers
    blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    blue_points = get_centers(blue_mask, min_area=100)

    # Green marker
    green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
    green_points = get_centers(green_mask, min_area=100)

    if len(blue_points) >= 2 and len(green_points) >= 1:
        # Sort blue points by Y position (top = forehead, bottom = chin)
        blue_points_sorted = sorted(blue_points, key=lambda p: p[1])
        forehead_pos = blue_points_sorted[0]
        chin_pos = blue_points_sorted[-1]
        green_pos = green_points[0]  # Assuming only one green marker

        # Calculate angle
        angle = calculate_angle(forehead_pos, green_pos, chin_pos)

        # Draw markers
        cv2.circle(resized, forehead_pos, 5, (255, 0, 0), -1)  # Blue
        cv2.circle(resized, chin_pos, 5, (255, 0, 0), -1)      # Blue
        cv2.circle(resized, green_pos, 5, (0, 255, 0), -1)     # Green

        # Draw lines
        cv2.line(resized, forehead_pos, green_pos, (0, 0, 0), 2)
        cv2.line(resized, green_pos, chin_pos, (0, 0, 0), 2)

        # Display angle
        cv2.putText(resized, f" Theta: {angle:.2f} and t: {time_stamp:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        angle = calculate_angle(forehead_pos, green_pos, chin_pos)
        results.append({"frame_idx": frame_idx, "t": time_stamp, "theta": angle})

    else:
        results.append({"frame_idx": frame_idx, "t": time_stamp, "theta": np.nan})
        cv2.putText(resized, "Markers missing", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("Jaw Tracking", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 

# Create DataFrame & save
df = pd.DataFrame(results)
df.to_csv("jaw_angle_dataset.csv", index=False)
# Define phase ranges
phases = {
    "Reference": (0, 512),
    "Max_Open": (513, 771),
    "Speech": (772, df["frame_idx"].max())
}

# Calculate velocity using central difference method
ang_vel = []
theta = df["theta"].values
for i in range(1, len(df)-1):
    vel = (theta[i+1] - theta[i-1]) / (2 * frame_duration)
    ang_vel.append(vel)

df["omega"] = [0] + ang_vel + [0]

# Calculate mean for each phase
phase_means = {}
for phase, (start, end) in phases.items():
    mean_angle = df.loc[(df["frame_idx"] >= start) & (df["frame_idx"] <= end), "theta"].mean()
    mean_velocity = df.loc[(df["frame_idx"] >= start) & (df["frame_idx"] <= end), "omega"].mean()
    phase_means[phase] = {"Mean_Angle": mean_angle, "Mean_Velocity": mean_velocity}

# Save results
df.to_csv("jaw_angle_dataset.csv", index=False)
pd.DataFrame.from_dict(phase_means, orient="index", columns=["Mean_Angle","Mean_Velocity"]).to_csv("phase_means.csv")

print("Frame-by-frame dataset saved as 'jaw_angle_dataset.csv'")
print("Phase-wise means saved as 'phase_means.csv'")
print("\nPhase Means:", phase_means)


