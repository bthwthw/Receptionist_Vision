import math
import cv2
import mediapipe as mp  

class FaceTracker:
    def __init__(self):
        print("[INFO] Loading MediaPipe Face Detection Module...")
        
        # Gọi thẳng mp.solutions như sách giáo khoa
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Pinhole Camera Intrinsics & Configs
        self.fx = 265.0075
        self.cx = 336.0675 
        self.left_eye_offset_deg = -2.5 
        
        self.height_ratio_threshold = 0.15

    def get_yaw_angle(self, x_center):
        angle_rad = math.atan((x_center - self.cx) / self.fx)
        return math.degrees(angle_rad) + self.left_eye_offset_deg

    def detect(self, frame):
        frame_height, frame_width, _ = frame.shape
        
        # MediaPipe yêu cầu RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        valid_targets = []
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                x1 = int(bboxC.xmin * frame_width)
                y1 = int(bboxC.ymin * frame_height)
                w = int(bboxC.width * frame_width)
                h = int(bboxC.height * frame_height)
                
                x2 = x1 + w
                y2 = y1 + h
                
                height_ratio = h / frame_height
                
                if height_ratio > self.height_ratio_threshold:
                    x_center = x1 + (w / 2.0)
                    valid_targets.append({
                        'box': (x1, y1, x2, y2),
                        'ratio': height_ratio,
                        'angle': self.get_yaw_angle(x_center)
                    })

        if valid_targets:
            return max(valid_targets, key=lambda t: t['ratio'])
        return None