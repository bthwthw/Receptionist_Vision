import cv2
import time
import os
from deepface import DeepFace

class FaceRecognizer:
    def __init__(self, db_path="database"):
        print("[INFO] Loading DeepFace Recognition Module (CPU Mode)...")
        self.db_path = db_path
        
        # Define sub-directories
        self.staff_path = os.path.join(self.db_path, "staff")
        self.vip_path = os.path.join(self.db_path, "vip")
        
        # Ensure database structure exists
        for path in [self.db_path, self.staff_path, self.vip_path]:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"[WARN] Created empty folder at '{path}'.")

        # Recognition States
        self.current_name = "Scanning..."
        self.current_role = "Unknown" 
        self.has_recognized = False

    def reset_state(self):
        """Reset internal state to allow fresh recognition for the next person."""
        self.current_name = "Scanning..."
        self.current_role = "Unknown"
        self.has_recognized = False

    def process(self, frame, target_box):
        """Perform one-time face recognition and categorize by folder."""
        x1, y1, x2, y2 = target_box
        
        if not self.has_recognized:
            margin = 20
            h, w = frame.shape[:2]
            cy1, cy2 = max(0, y1 - margin), min(h, y2 + margin)
            cx1, cx2 = max(0, x1 - margin), min(w, x2 + margin)
            
            face_crop = frame[cy1:cy2, cx1:cx2]
            
            # Count images across all subfolders
            total_images = sum([len(files) for r, d, files in os.walk(self.db_path) if not any(f.endswith('.pkl') for f in files)])
            
            if face_crop.size > 0 and total_images > 0:
                try:
                    # Run identification using the most robust metric for Facenet
                    dfs = DeepFace.find(
                        img_path=face_crop, 
                        db_path=self.db_path, 
                        model_name="Facenet", 
                        distance_metric="euclidean_l2", 
                        enforce_detection=False, 
                        silent=True 
                    )
                    
                    if len(dfs) > 0 and not dfs[0].empty:
                        # Get the best match distance (lowest score = best match)
                        match_distance = dfs[0]['distance'][0] if 'distance' in dfs[0].columns else dfs[0].iloc[0, -1]
                        
                        # --- DEBUG: See the actual score in the terminal ---
                        # Use this value to fine-tune your threshold.
                        print(f"[DEBUG] Best match distance found: {match_distance:.4f}")
                        
                        # Threshold Logic: Adjust 0.65 based on the debug values you see
                        if match_distance < 0.65:
                            matched_path = dfs[0]['identity'][0]
                            
                            # 1. Extract Name
                            filename = os.path.basename(matched_path)
                            self.current_name = os.path.splitext(filename)[0]
                            
                            # 2. Extract Role based on sub-folder location
                            normalized_path = matched_path.replace('\\', '/')
                            if "/vip/" in normalized_path.lower():
                                self.current_role = "VIP"
                            elif "/staff/" in normalized_path.lower():
                                self.current_role = "Staff"
                            else:
                                self.current_role = "Guest"
                        else:
                            # Distance too high, person is likely not in the database
                            self.current_name = "Unknown"
                            self.current_role = "Guest"
                            
                    else:
                        print("[DEBUG] DeepFace returned NO results. The face is completely different or the .pkl file is outdated.")
                        self.current_name = "Unknown"
                        self.current_role = "Guest"
                        
                except Exception as e:
                    print(f"[ERROR] Recognition logic failure: {e}")
                    self.current_name = "Error"
                    self.current_role = "Guest"
            else:
                self.current_name = "Empty DB"
                self.current_role = "Guest"
                
            # Lock recognition to save CPU cycles
            self.has_recognized = True

        # --- UI Drawing ---
        if self.current_role == "VIP":
            color = (0, 215, 255) 
        elif self.current_role == "Staff":
            color = (255, 144, 30) 
        else:
            color = (150, 150, 150) 
            
        tag_text = f"[{self.current_role}] {self.current_name}"
        cv2.putText(frame, tag_text, (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        return frame