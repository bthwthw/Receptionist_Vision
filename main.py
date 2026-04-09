import cv2
import time

from modules.tracker import FaceTracker
from modules.face_id import FaceRecognizer
from MQTT.angle_publisher import AnglePublisher, get_topic

class VisionNodeManager:
    def __init__(self):
        # 1. Initialize Core Vision Modules
        self.tracker = FaceTracker()
        self.face_id = FaceRecognizer()
        
        # 2. System State and Timer Initialization
        self.status = "Idle"           # Current mode of the robot
        self.first_seen_time = 0.0     # Timestamp when a person is first detected
        self.last_seen_time = 0.0      # Timestamp of the most recent detection
        self.trigger_delay = 2.0       # Seconds to wait before trusting the detection and moving
        self.absence_threshold = 3.0   # Seconds to wait before assuming the person left
        self.deadband_threshold = 5.0  # Minimum angle (degrees) required to trigger a rotation
        
        # 3. Hardware / Camera Configuration
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 4. MQTT
        self._angle_publisher = None
        self.setup_mqtt()

    def setup_mqtt(self):
        """Initializes the MQTT publisher to send rotation commands to the hardware layer."""
        try:
            self._angle_publisher = AnglePublisher()
            self._angle_publisher.topic = get_topic("xoay")
        except Exception as e:
            print(f"[WARN] MQTT setup failed: {e}")

    def publish_rotation(self, angle_deg):
        """Formats and publishes the target rotation angle to the MQTT broker."""
        if self._angle_publisher is None: return
        try:
            target_angle = float(round(angle_deg, 3))
            self._angle_publisher.publish_angle({"angle": target_angle})
            print(f"\n[MQTT] === COMMAND FIRED: Rotate by {target_angle} degrees ===")
        except Exception as e:
            print(f"[ERROR] Publish failed: {e}")

    def run(self):
        """Main execution loop that handles video capture, logic routing, and UI rendering."""
        print("[INFO] System Started...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
                
            # Extract the left eye frame from the ZED stereo image
            left_frame = frame[:, :frame.shape[1]//2]
            current_time = time.time()
            
            # Continuously scan for faces to update the last_seen_time (prevents command spam)
            target = self.tracker.detect(left_frame)
            
            if target:
                # Update the timestamp since a face is currently visible
                self.last_seen_time = current_time
                bx1, by1, bx2, by2 = target['box']
                
                # --- STATE MACHINE ROUTING ---
                if self.status == "Idle":
                    if self.first_seen_time == 0.0:
                        self.first_seen_time = current_time
                        print("[INFO] Target acquired. Validating (2s)...")
                        
                    elif (current_time - self.first_seen_time) >= self.trigger_delay:
                        # 1. Check if we need to rotate
                        if abs(target['angle']) > self.deadband_threshold:
                            self.publish_rotation(target['angle'])
                        else:
                            print("[INFO] Target already centered. No rotation needed.")
                            
                        # 2. ALWAYS switch to Greeting to start Face ID
                        self.status = "Greeting"
                        self.first_seen_time = 0.0
                            
                    # UI Rendering for Idle State
                    color = (0, 165, 255) if self.first_seen_time > 0 else (0, 255, 0)
                    cv2.rectangle(left_frame, (bx1, by1), (bx2, by2), color, 3)
                    cv2.putText(left_frame, f"Angle: {target['angle']:.1f} deg", (bx1, by1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                elif self.status == "Greeting":
                    # Activate the DeepFace recognition module while interacting
                    left_frame = self.face_id.process(left_frame, target['box'])
                    cv2.rectangle(left_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)

            else:                
                # If we were validating someone and they disappeared, cancel the timer
                if self.first_seen_time > 0.0:
                    self.first_seen_time = 0.0
                    print("[INFO] Target lost during validation. Resetting.")
                    
                # If we were greeting someone and they left
                if self.status == "Greeting":
                    if (current_time - self.last_seen_time) > self.absence_threshold:
                        print(f"[INFO] Target absent for {self.absence_threshold}s. Resetting to Idle.")
                        self.status = "Idle"
                        # Ensure the Face ID module forgets the previous person
                        # self.face_id.reset_state()

            # Global UI Rendering
            cv2.putText(left_frame, f"STATUS: {self.status}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Receptionist Robot Node", left_frame)
            
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Resource cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = VisionNodeManager()
    node.run()