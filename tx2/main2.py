import pyzed.sl as sl
import cv2
import numpy as np
import csv # Thư viện xuất file
import time

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

    with open('tracking_data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Ghi tiêu đề cột
            writer.writerow(['Timestamp', 'ID', 'X', 'Y', 'Z', 'Distance'])

            while True:
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    frame = image.get_data()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

                    zed.retrieve_objects(objects, obj_runtime_param)

                    for obj in objects.object_list:
                        # Lấy tọa độ đầy đủ
                        pos = obj.position
                        x, y, z_coord = pos[0], pos[1], pos[2]
                        dist = np.linalg.norm(pos)
                        timestamp = time.time()

                        # Ghi vào file CSV
                        writer.writerow([timestamp, obj.id, x, y, z_coord, dist])

                        # Vẽ lên màn hình để kiểm tra
                        pt1 = (int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1]))
                        pt2 = (int(obj.bounding_box_2d[2][0]), int(obj.bounding_box_2d[2][1]))
                        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                        
                        label = f"ID:{obj.id} X:{x:.2f} Y:{y:.2f} Z:{z_coord:.2f}"
                        cv2.putText(frame, label, (pt1[0], pt1[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    cv2.imshow("ZED 3D Localization", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    # Đóng mọi thứ
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()

if __name__ == "__main__":
    main()