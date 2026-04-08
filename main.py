import cv2
import math
import numpy as np
from ultralytics import YOLO
import time

from MQTT.angle_publisher import AnglePublisher, get_topic

class ReceptionistVision:
    def __init__(self, model_path, cam_id=0):
        print("[INFO] Loading OpenVINO Model...")
        self.model = YOLO(model_path, task='detect')

        # Pinhole Camera Intrinsics
        self.fx = 265.0075
        self.cx = 336.0675 
        
        # System State
        self.status = "Idle"
        
        # Timeout tracking for Absence (Switch to Idle)
        self.last_seen_time = 0.0
        self.absence_threshold = 3.0 # Wait 3 seconds of empty frame to reset to Idle
        
        # Validation tracking for Presence (Switch to Greeting)
        self.first_seen_time = 0.0
        self.trigger_delay = 2.0 # Wait 2 continuous seconds before sending MQTT
        
        # Camera Setup (ZED2 Side-by-side)
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Hardware Compensation (Parallax Error)
        # Shift angle by approx -2.5 degrees to compensate for the left lens offset
        self.left_eye_offset_deg = -2.5 
        # Deadband threshold to ignore small deviations (e.g., <5 degrees)
        self.deadband_threshold = 5.0
        # Minimum height ratio to filter out distant/small detections (tune based on environment)
        self.height_ratio_threshold = 0.5 
        self._angle_publisher = None 

    def get_yaw_angle(self, x_center):
        """Calculate the horizontal deviation angle and apply parallax compensation."""
        angle_rad = math.atan((x_center - self.cx) / self.fx)
        raw_angle_deg = math.degrees(angle_rad)
        
        # Apply offset to align with the true center of the robot
        compensated_angle = raw_angle_deg + self.left_eye_offset_deg
        return compensated_angle

    def publish_mqtt(self, angle_deg):
        """Publish the target angle via MQTT."""
        try:
            if self._angle_publisher is None:
                self._angle_publisher = AnglePublisher()
            
            # Keep the signed value for relative rotation
            target_angle_val = float(round(angle_deg, 3))
            
            self._angle_publisher.publish_angle({"angle": target_angle_val})
            print(f"\n[MQTT] === COMMAND FIRED: Rotate by {target_angle_val} degrees ===")
            
        except Exception as e:
            print(f"[ERROR] MQTT publish failed: {e}")

    def run(self):
        print("[INFO] Vision System Started...")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
                
            # Extract Left Eye
            height, width, _ = frame.shape
            left_frame = frame[:, :width//2]
            frame_height = left_frame.shape[0]
            
            # Object Detection
            results = self.model.predict(source=left_frame, classes=[0], conf=0.5, verbose=False)
            
            valid_targets = []
            
            # Parse results
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_height = y2 - y1
                height_ratio = box_height / frame_height
                
                if height_ratio > self.height_ratio_threshold:
                    x_center = (x1 + x2) / 2.0
                    angle = self.get_yaw_angle(x_center)
                    valid_targets.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'ratio': height_ratio,
                        'angle': angle
                    })
            
            # Logic: If a valid target is currently in the frame
            if valid_targets:
                self.last_seen_time = time.time() 
                closest_person = max(valid_targets, key=lambda t: t['ratio'])
                
                # State Machine Logic for "Idle" -> "Greeting"
                if self.status == "Idle":
                    # Mark the first time we see someone
                    if self.first_seen_time == 0.0:
                        self.first_seen_time = time.time()
                        print("[INFO] Target acquired. Validating presence (1s)...")
                        
                    # Check if 1 second has passed continuously
                    elif ((time.time() - self.first_seen_time) >= self.trigger_delay) and abs(closest_person['angle']) > self.deadband_threshold:
                        self.publish_mqtt(closest_person['angle'])
                        self.status = "Greeting"
                        self.first_seen_time = 0.0 # Reset validation timer
                
                # Visuals
                bx1, by1, bx2, by2 = closest_person['box']
                color = (0, 255, 0) if self.status == "Greeting" else (0, 165, 255) # Orange for validation, Green for greeting
                cv2.rectangle(left_frame, (bx1, by1), (bx2, by2), color, 3)
                
                status_text = f"STATUS: {self.status}"
                if self.status == "Idle" and self.first_seen_time > 0:
                    status_text += f" (Validating...)"
                    
                cv2.putText(left_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(left_frame, f"Angle: {closest_person['angle']:.1f} deg", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Logic: If NO target is seen
            else:
                # Reset the validation timer immediately if target is lost before 1 second
                if self.first_seen_time > 0.0:
                    print("[INFO] Target lost during validation. Timer reset.")
                    self.first_seen_time = 0.0
                
                # Check absence timeout to return to Idle
                if self.status == "Greeting":
                    elapsed_absence = time.time() - self.last_seen_time
                    if elapsed_absence > self.absence_threshold:
                        print(f"[INFO] Target absent for {self.absence_threshold}s. Resetting to Idle.")
                        self.status = "Idle"

            cv2.imshow("Receptionist Robot - Perception Layer", left_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # yolo export model=yolov8n.pt format=openvino
    vision_module = ReceptionistVision(model_path="yolov8n_openvino_model/")
    vision_module.run()