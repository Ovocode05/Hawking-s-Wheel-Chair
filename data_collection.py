# import cv2
# import numpy as np
# import pandas as pd
# import os
# import time
# import threading
# from collections import deque
# import sounddevice as sd
# from scipy.io.wavfile import write as write_wav

# # ----------------------------
# # 1. CONFIGURATION
# # ----------------------------
# # Video Settings
# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 720
# FPS = 30  # Desired FPS for recording

# # Audio Settings
# SAMPLE_RATE = 44100  # Standard audio sample rate
# CHANNELS = 1         # Mono audio

# # Marker Detection Settings
# LOWER_BLUE = (90, 50, 50)
# UPPER_BLUE = (130, 255, 255)
# LOWER_GREEN = (40, 50, 50)
# UPPER_GREEN = (80, 255, 255)
# MIN_CONTOUR_AREA = 100

# # Motion Detection Settings
# MOVEMENT_THRESHOLD = 0.5  # StDev of angle. Lower means more sensitive to stopping.
# STOP_DELAY_FRAMES = 90    # How many frames of no movement to wait before stopping (e.g., 90 frames = 3 seconds at 30fps)
# HISTORY_LEN = 15          # Number of recent angles to check for movement

# # File Paths
# BASE_RECORDING_FOLDER = "Recordings"

# # Global variables for threading and state management
# is_recording = False
# stop_thread = False
# audio_frames = []

# # ----------------------------
# # 2. HELPER FUNCTIONS
# # ----------------------------
# def get_centers(mask, min_area=100):
#     """Finds the centers of contours in a binary mask."""
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     centers = []
#     for cnt in contours:
#         if cv2.contourArea(cnt) > min_area:
#             M = cv2.moments(cnt)
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 centers.append((cx, cy))
#     return centers

# def calculate_angle(a, b, c):
#     """Calculates angle ABC in degrees."""
#     ba = np.array(a) - np.array(b)
#     bc = np.array(c) - np.array(b)
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#     return np.degrees(angle)

# def audio_callback(indata, frames, time, status):
#     """This is called (from a separate thread) for each audio block."""
#     if status:
#         print(status)
#     if is_recording:
#         audio_frames.append(indata.copy())

# def start_audio_stream():
#     """Starts the non-blocking audio stream."""
#     global stream
#     stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
#     stream.start()
#     print("🎤 Audio stream started.")

# def save_recording(output_folder, video_writer, results_data):
#     """Saves video, audio, and data, and resets state."""
#     global is_recording, audio_frames

#     print(f"\n⏹️ Stopping recording...")
#     is_recording = False
#     time.sleep(0.5) # Give threads a moment to stop

#     # --- Save Video ---
#     video_writer.release()
#     print(f"✅ Video saved to {output_folder}")

#     # --- Save Audio ---
#     if audio_frames:
#         audio_path = os.path.join(output_folder, "audio.wav")
#         recording = np.concatenate(audio_frames, axis=0)
#         write_wav(audio_path, SAMPLE_RATE, recording)
#         print(f"✅ Audio saved to {audio_path}")
#     audio_frames = [] # Clear for next recording

#     # --- Process and Save Data ---
#     df = pd.DataFrame(results_data)
#     if not df.empty and 'theta' in df.columns:
#         frame_duration = 1 / FPS
#         # Velocity
#         omega = np.gradient(df['theta'].fillna(method='ffill').fillna(method='bfill'), df['t'])
#         df['omega'] = omega
        
#         # Acceleration
#         alpha = np.gradient(df['omega'], df['t'])
#         df['alpha'] = alpha
        
#         data_path = os.path.join(output_folder, "kinematic_data.csv")
#         df.to_csv(data_path, index=False)
#         print(f"✅ Kinematic data saved to {data_path}")

#     print("-" * 30)


# # ----------------------------
# # 3. MAIN APPLICATION LOGIC
# # ----------------------------
# def main():
#     global is_recording

#     # --- Setup ---
#     session_name = input("Enter a name for this recording session (e.g., 'hello_world'): ").strip()
#     session_folder = os.path.join(BASE_RECORDING_FOLDER, session_name)
#     os.makedirs(session_folder, exist_ok=True)
#     sample_count = 0

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return
        
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#     cap.set(cv2.CAP_PROP_FPS, FPS)
    
#     start_audio_stream()

#     # --- State Variables for each recording ---
#     video_out = None
#     results = []
#     angle_history = deque(maxlen=HISTORY_LEN)
#     no_movement_counter = 0

#     print("\n--- Controls ---")
#     print(" 's' - Start a new recording sample")
#     print(" 'q' - Quit the application")
#     print("------------------\n")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # --- Marker Detection and Calculation ---
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
#         green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

#         blue_points = get_centers(blue_mask, min_area=MIN_CONTOUR_AREA)
#         green_points = get_centers(green_mask, min_area=MIN_CONTOUR_AREA)
        
#         current_angle = np.nan
#         if len(blue_points) >= 2 and len(green_points) >= 1:
#             blue_points_sorted = sorted(blue_points, key=lambda p: p[1])
#             forehead_pos, chin_pos = blue_points_sorted[0], blue_points_sorted[-1]
#             green_pos = green_points[0]
#             current_angle = calculate_angle(forehead_pos, green_pos, chin_pos)

#             # Draw annotations
#             cv2.circle(frame, forehead_pos, 7, (255, 0, 0), -1)
#             cv2.circle(frame, chin_pos, 7, (255, 0, 0), -1)
#             cv2.circle(frame, green_pos, 7, (0, 255, 0), -1)
#             cv2.line(frame, forehead_pos, green_pos, (0, 0, 0), 2)
#             cv2.line(frame, green_pos, chin_pos, (0, 0, 0), 2)
#             cv2.putText(frame, f"Angle: {current_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         # --- Recording Logic ---
#         if is_recording:
#             # Append data for the current frame
#             frame_idx = len(results)
#             time_stamp = frame_idx / FPS
#             results.append({
#                 "frame_idx": frame_idx,
#                 "t": time_stamp,
#                 "theta": current_angle
#             })

#             # Write frame to video file
#             video_out.write(frame)

#             # Check for movement
#             if not np.isnan(current_angle):
#                 angle_history.append(current_angle)
#                 if len(angle_history) == HISTORY_LEN:
#                     # If standard deviation is low, increment no movement counter
#                     if np.std(angle_history) < MOVEMENT_THRESHOLD:
#                         no_movement_counter += 1
#                     else: # Reset counter if movement is detected
#                         no_movement_counter = 0
            
#             # If no movement for a while, stop recording
#             if no_movement_counter >= STOP_DELAY_FRAMES:
#                 save_recording(output_folder, video_out, results)
#                 # Reset for next recording
#                 results = []
#                 angle_history.clear()
#                 no_movement_counter = 0

#         # --- UI and Controls ---
#         key = cv2.waitKey(1) & 0xFF
        
#         if key == ord('q'):
#             if is_recording:
#                 save_recording(output_folder, video_out, results)
#             break
        
#         elif key == ord('s') and not is_recording:
#             # --- Start a new recording ---
#             is_recording = True
#             sample_count += 1
#             output_folder = os.path.join(session_folder, f"sample_{sample_count}")
#             os.makedirs(output_folder, exist_ok=True)
            
#             # Setup video writer
#             video_path = os.path.join(output_folder, "video.mp4")
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             video_out = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            
#             # Reset state
#             results = []
#             audio_frames = [] # Clear audio buffer
#             angle_history.clear()
#             no_movement_counter = 0
            
#             print(f"\n▶️ Started recording for Sample {sample_count}")
#             print(f"   Saving to: {output_folder}")

#         # Display status on screen
#         status_text = "RECORDING" if is_recording else "IDLE (Press 's' to start)"
#         color = (0, 255, 0) if is_recording else (255, 255, 0)
#         cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

#         cv2.imshow("Jaw Movement Data Collector", frame)

#     # --- Cleanup ---
#     cap.release()
#     stream.stop()
#     stream.close()
#     cv2.destroyAllWindows()
#     print("Application closed.")


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import pandas as pd
import os
import time
import threading
from collections import deque
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

# ----------------------------
# 1. CONFIGURATION
# ----------------------------
# Video Settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Audio Settings
SAMPLE_RATE = 44100
CHANNELS = 1

# --- MODIFIED: Marker Detection Settings ---
# Two YELLOW markers for forehead and chin
LOWER_YELLOW = (20, 100, 100)
UPPER_YELLOW = (40, 255, 255)

# One ORANGE marker for the jaw joint
# NOTE: Orange is tricky. You will likely need to tune these values.
LOWER_ORANGE = (5, 120, 120)
UPPER_ORANGE = (20, 255, 255)

MIN_CONTOUR_AREA = 100
# --- END MODIFICATION ---

# Motion Detection Settings
MOVEMENT_THRESHOLD = 0.5
STOP_DELAY_FRAMES = 60
HISTORY_LEN = 15

# File Paths
BASE_RECORDING_FOLDER = "Recordings_YellowOrange"

# Global variables
is_recording = False
audio_frames = []
stream = None

# ----------------------------
# 2. HELPER FUNCTIONS (Unchanged)
# ----------------------------
def get_centers(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
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

def audio_callback(indata, frames, time, status):
    if status: print(status)
    if is_recording: audio_frames.append(indata.copy())

def start_audio_stream():
    global stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback)
    stream.start()
    print("🎤 Audio stream started.")

def save_recording(output_folder, video_writer, results_data):
    global is_recording, audio_frames
    is_recording = False
    print(f"\n⏹️ Stopping recording...")
    time.sleep(0.5)

    if video_writer:
        video_writer.release()
        print(f"✅ Video saved to {output_folder}")

    if audio_frames:
        audio_path = os.path.join(output_folder, "audio.wav")
        recording = np.concatenate(audio_frames, axis=0)
        write_wav(audio_path, SAMPLE_RATE, recording)
        print(f"✅ Audio saved to {audio_path}")
    audio_frames = []

    df = pd.DataFrame(results_data)
    if not df.empty and 'theta' in df.columns:
        omega = np.gradient(df['theta'].interpolate(method='linear'), df['t'])
        df['omega'] = omega
        alpha = np.gradient(df['omega'], df['t'])
        df['alpha'] = alpha
        data_path = os.path.join(output_folder, "kinematic_data.csv")
        df.to_csv(data_path, index=False)
        print(f"✅ Kinematic data saved to {data_path}")
    print("-" * 30)

# ----------------------------
# 3. MAIN APPLICATION LOGIC
# ----------------------------
def main():
    global is_recording
    session_name = input("Enter a name for this recording session: ").strip()
    session_folder = os.path.join(BASE_RECORDING_FOLDER, session_name)
    os.makedirs(session_folder, exist_ok=True)
    sample_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    start_audio_stream()

    video_out, results, angle_history, no_movement_counter = None, [], deque(maxlen=HISTORY_LEN), 0

    print("\n--- Controls ---\n 's' - Start\n 'q' - Quit\n------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- MODIFIED: Marker Detection and Calculation ---
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR_HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for YELLOW and ORANGE
        yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
        orange_mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

        # Get centers for YELLOW and ORANGE points
        yellow_points = get_centers(yellow_mask, min_area=MIN_CONTOUR_AREA)
        orange_points = get_centers(orange_mask, min_area=MIN_CONTOUR_AREA)
        
        current_angle = np.nan
        # Check for 2 yellow points and 1 orange point
        if len(yellow_points) >= 2 and len(orange_points) >= 1:
            # Sort yellow points by y-coordinate to find top and bottom
            yellow_points_sorted = sorted(yellow_points, key=lambda p: p[1])
            forehead_pos, chin_pos = yellow_points_sorted[0], yellow_points_sorted[-1]
            
            # Use the first detected orange point as the jaw position
            jaw_pos = orange_points[0]
            current_angle = calculate_angle(forehead_pos, jaw_pos, chin_pos)

            # Draw annotations
            # BGR color for yellow is (0, 255, 255)
            cv2.circle(frame, forehead_pos, 7, (0, 255, 255), -1)
            cv2.circle(frame, chin_pos, 7, (0, 255, 255), -1)
            # BGR color for orange is (0, 165, 255)
            cv2.circle(frame, jaw_pos, 7, (0, 165, 255), -1)
            
            cv2.line(frame, forehead_pos, jaw_pos, (0, 0, 0), 2)
            cv2.line(frame, jaw_pos, chin_pos, (0, 0, 0), 2)
            cv2.putText(frame, f"Angle: {current_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # --- END MODIFICATION ---

        if is_recording:
            frame_idx = len(results)
            time_stamp = frame_idx / FPS
            results.append({"frame_idx": frame_idx, "t": time_stamp, "theta": current_angle})
            video_out.write(frame)

            if not np.isnan(current_angle):
                angle_history.append(current_angle)
                if len(angle_history) == HISTORY_LEN and np.std(angle_history) < MOVEMENT_THRESHOLD:
                    no_movement_counter += 1
                else:
                    no_movement_counter = 0
            
            if no_movement_counter >= STOP_DELAY_FRAMES:
                save_recording(output_folder, video_out, results)
                results, angle_history, no_movement_counter = [], deque(maxlen=HISTORY_LEN), 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if is_recording: save_recording(output_folder, video_out, results)
            break
        elif key == ord('s') and not is_recording:
            is_recording = True
            sample_count += 1
            output_folder = os.path.join(session_folder, f"sample_{sample_count}")
            os.makedirs(output_folder, exist_ok=True)
            
            video_path = os.path.join(output_folder, "video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_out = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            
            results, audio_frames, angle_history, no_movement_counter = [], [], deque(maxlen=HISTORY_LEN), 0
            print(f"\n▶️ Started recording for Sample {sample_count} -> {output_folder}")

        status_text = "RECORDING" if is_recording else "IDLE (Press 's' to start)"
        color = (0, 0, 255) if is_recording else (255, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow("Jaw Movement Data Collector", frame)

    cap.release()
    if stream:
        stream.stop()
        stream.close()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()