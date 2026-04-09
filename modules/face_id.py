import cv2

class FaceRecognizer:
    def __init__(self):
        # Khởi tạo model FaceID ở đây sau này
        pass

    def process(self, frame, target_box):
        """Chạy nhận diện khuôn mặt trong bounding box của người đó"""
        cv2.putText(frame, "Module: Face ID / Recognition Running...", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Sau này sẽ return thêm tên người nhận diện được
        return frame