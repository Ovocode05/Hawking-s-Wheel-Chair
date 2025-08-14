import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
# Ensure folders exist
# ----------------------------
os.makedirs("Data/Bahaar", exist_ok=True)
os.makedirs("Graphs", exist_ok=True)

# ----------------------------
# Video Processing
# ----------------------------
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise ValueError("FPS is zero. Check camera/video file.")
frame_duration = 1 / fps

results = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    time_stamp = frame_idx * frame_duration

    # Resize
    resized = cv2.resize(frame, (1250, 700), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Blue markers
    blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    blue_points = get_centers(blue_mask, min_area=100)

    # Green marker
    green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
    green_points = get_centers(green_mask, min_area=100)

    if len(blue_points) >= 2 and len(green_points) >= 1:
        blue_points_sorted = sorted(blue_points, key=lambda p: p[1])
        forehead_pos = blue_points_sorted[0]
        chin_pos = blue_points_sorted[-1]
        green_pos = green_points[0]

        angle = calculate_angle(forehead_pos, green_pos, chin_pos)
        vertical_disp = chin_pos[1] - forehead_pos[1]   # Δy
        horizontal_disp = chin_pos[0] - forehead_pos[0]   # Δx

        # Draw
        cv2.circle(resized, forehead_pos, 5, (255, 0, 0), -1)
        cv2.circle(resized, chin_pos, 5, (255, 0, 0), -1)
        cv2.circle(resized, green_pos, 5, (0, 255, 0), -1)
        cv2.line(resized, forehead_pos, green_pos, (0, 0, 0), 2)
        cv2.line(resized, green_pos, chin_pos, (0, 0, 0), 2)
        cv2.putText(resized, f"Theta: {angle:.2f} | t: {time_stamp:.2f}s | Δy: {vertical_disp} | Δx: {horizontal_disp}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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

# ----------------------------
# Data Processing
# ----------------------------
df = pd.DataFrame(results)

# Velocity
ang_vel = []
theta = df["theta"].values
for i in range(1, len(df) - 1):
    vel = (theta[i + 1] - theta[i - 1]) / (2 * frame_duration)
    ang_vel.append(vel)
df["omega"] = [0] + ang_vel + [0]

# Acceleration
ang_acc = []
for i in range(1, len(df) - 1):
    acc = (theta[i + 1] - 2 * theta[i] + theta[i - 1]) / (frame_duration ** 2)
    ang_acc.append(acc)
df["alpha"] = [0] + ang_acc + [0]

# ----------------------------
# Phase Means Example
# ----------------------------
phases = {
    "Reference": (0, 150),
    "Max_Open": (151, 300),
}
phase_means = {}
for phase, (start, end) in phases.items():
    mean_angle = df.loc[(df["frame_idx"] >= start) & (df["frame_idx"] <= end), "theta"].mean()
    mean_velocity = df.loc[(df["frame_idx"] >= start) & (df["frame_idx"] <= end), "omega"].mean()
    mean_acceleration = df.loc[(df["frame_idx"] >= start) & (df["frame_idx"] <= end), "alpha"].mean()
    phase_means[phase] = {
        "Mean_Angle": mean_angle,
        "Mean_Velocity": mean_velocity,
        "Mean_Acceleration": mean_acceleration
    }

# ----------------------------
# Save Outputs
# ----------------------------
df.to_csv("Data/Bahaar/baharasa.csv", index=False)
pd.DataFrame.from_dict(phase_means, orient="index").to_csv("Data/Bahaar/phase_meanssass.csv")

# ----------------------------
# Plot and Save Graph
# ----------------------------
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df['t'], df['theta'], label='Angle (theta)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Jaw Angle vs Time')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df['t'], df['omega'], label='Angular Velocity (omega)', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (deg/s)')
plt.title('Angular Velocity vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('Graphs/angle_vs_time.png')
plt.show()

print("\nPhase Means:", phase_means)
while True:
    choice = input(f"Do you want to save the recording'? (y/n): ").strip().lower()
    if choice == 'y':
        print("✅ Recording saved.")
        break
    elif choice == 'n':
        os.remove("Graphs/angle_vs_time.png")
        os.remove("Data/Bahaar/baharasa.csv")
        os.remove("Data/Bahaar/phase_meanssass.csv")
        print("🗑️ Recording deleted.")
        break
    else:
        print("Invalid input, please enter 'y' or 'n'.")
