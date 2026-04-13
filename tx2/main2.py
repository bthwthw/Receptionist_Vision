import pyzed.sl as sl
import cv2
import numpy as np

def main():
    zed = sl.Camera()

    # Cấu hình khởi tạo
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 30
    init_params.sdk_verbose = True
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Để lấy tọa độ 3D chính xác nhất
    init_params.coordinate_units = sl.UNIT.METER 
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Không mở được camera, check lại cổng USB 3.0!")
        return
    
    # 2. BẮT BUỘC: Bật Positional Tracking trước
    # Điều này giúp ZED hiểu được sự di chuyển của camera để giữ ID vật thể ổn định
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # positional_tracking_parameters.set_as_static = True # Nếu camera đứng yên trên chân đế, hãy bỏ comment dòng này
    zed.enable_positional_tracking(positional_tracking_parameters)

    # 3. Sau đó mới bật Object Detection
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True # Giờ thì dòng này mới chạy được nè
    obj_param.image_sync = True
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 60

    image = sl.Mat()
    objects = sl.Objects()

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # 1. Lấy ảnh trái để hiển thị
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data() # Chuyển sang numpy để dùng cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) # ZED mặc định là RGBA

            # 2. Lấy dữ liệu Object
            zed.retrieve_objects(objects, obj_runtime_param)

            for obj in objects.object_list:
                # Tọa độ 2D (Pixel)
                box = obj.bounding_box_2d
                pt1 = (int(box[0][0]), int(box[0][1]))
                pt2 = (int(box[2][0]), int(box[2][1]))

                # Tọa độ 3D (Spatial Localization) - THỨ WEBCAM KHÔNG CÓ
                # position[0]: X (phải/trái), [1]: Y (lên/xuống), [2]: Z (khoảng cách tới camera)
                pos = obj.position 
                dist = np.linalg.norm(pos) # Khoảng cách thực tế (Euclidean distance)

                # Vẽ lên OpenCV
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                info = f"ID: {obj.id} | Z: {pos[2]:.2f}m | Dist: {dist:.2f}m"
                cv2.putText(frame, info, (pt1[0], pt1[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.imshow("ZED 3.4.2 - 3D Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.disable_object_detection()
    zed.close()

if __name__ == "__main__":
    main()