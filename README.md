# Autonomous-Car
DEVELOPMENT OF DRIVING-ASSISTED SYSTEM FOR AUTONOMOUS CAR

**Installation:** Open Terminal and do above step

**STEP 1: CREATE WORKSPACE** 
```
cd ~
mkdir NCKH_workspace
cd NCKH_workspace
mkdir KOT3_ws
cd KOT3_ws
mkdir src
```
**STEP 2: CONNECT WITH REMOTE REPO** 
```
cd src
git init 
git remote add origin https://github.com/CEK19/Autonomous-Car.git
git pull origin main
```
**STEP 3: BUILD WORKSPACE**
```
cd ~/NCKH_workspace/KOT3_ws/
catkin_make
```

**STEP 4: RUN PROJECT**
```
DO SOMETHING HERE
```

---
**Source structure**:   
-  **build:** contains script to build workspace ROS
-  **dataset_kitty:** folder contains dataset to train/test
-  **documentation:** contains software documentations & more information
-  **kot3_pkg:** contains script & folder relative to ROS
-  **report:** contains result of testing
-  **src:** contains mains algorithms script (don't have ROS script here)
---
**Contribute Project Rule:**

-**Commit message:** *"#ISSUE_NUMBER date: detail changing of this commit"* => **Example:** *"#4 20-08-2022: changing folder structure for understandable purpose"*

**Authors**:
- Nguyen Trong Nhan - 1914446
- Nguyen Duy Thinh - 1915313
- Le Hoang Minh Tu - 1915812

---
# Guideline:

---

**Module né vật cản**
- Các cách tiếp cận:
- Hướng tiếp cận đang được sử dụng:

---

**Module nhận diện làn đường**
- Các cách tiếp cận:
- Hướng tiếp cận đang được sử dụng:

---

**Module nhận diện và phân loại biển báo giao thông**
- Các cách tiếp cận: Có 2 cách tiếp cận chính gồm:
  - [1] OpenCV + CNN: Sử dụng các kĩ thuật Computer Vision để định ra vị trí sau đó cắt biển báo tại vùng boundary box đó để đưa vào CNN. Tuy nhiên thì do là thuật toán thị giác máy tính nên rất dễ bị ảnh hưởng bởi ánh sáng và không xử lí được nhiều trường hợp khác nhau.
  - [2] YOLOV8 + CNN: YOLOV8 được sử dụng trong việc Object Detection, tức là chỉ định ra vị trí của các biển báo trên hình và đóng khung chúng lại. Sau đó 
- Hướng tiếp cận đang được sử dụng: Cách [2] YOLOV8 + CNN. Chi tiết sẽ được mô tả bên dưới.

**Bước 1: [YOLOV8] Cài đặt thư viện**: 
```
pip install ultralytics
```

**Bước 2: [YOLOV8] Lựa chọn mô hình phù hợp:** 

Ở đây hiện đang có 4 loại Detection, Segmentation, Classification, Pose. Nhưng mục đích chính của YOLOV8 được sử dụng trong bài toán này là xác định được vị trí của biển báo nên ta sẽ sử dụng _YOLOV8n.pt_ để optimize thời gian xử lí. Vì ở bước này chưa cần độ chính xác quá cao.

**Bước 3: [YOLOV8] Chụp hình thực tế để thu thập dữ liệu**

**Bước 4: [YOLOV8] Gán nhãn dữ liệu:**

Truy cập https://www.makesense.ai/ > Đánh nhãn > Xuất ở định dạng YOLO XML > Download

**Bước 5: [YOLOV8] Huấn luyện mô hình:**
Tổng quan các bước gồm: Config file > Huấn luyện > Lưu model dưới định dạng "anyNameModel.pt"

**1. Tạo folder train và validation:**
   - Train:
     - /PATH_TO_TRAIN/train/images (trong đây chưa các ảnh train *.jpg)
     - /PATH_TO_TRAIN/train/labels (trong đây chưa các file train labels *.txt)
   - Validation:
     - /PATH_TO_VAL/val/images (trong đây chưa các ảnh validation *.jpg)
     - /PATH_TO_VAL/val/labels (trong đây chưa các file validation labels *.txt)
**2. Chỉnh sửa file configuration:**
```config.yaml
train: /PATH_TO_TRAIN/train/images
val: /PATH_TO_VALIDATION/val/images
names: 
  0: sign
  // 1: another_thing1 (if need) -> new label
  // 2: another_thing (if need) -> new label 
```
**3. Huấn luyện: Tham khảo file sau main.ipynb:**
[Reference Training Folder](https://drive.google.com/drive/folders/1odlIC2L1V09jNq_5MxbbNF4dSDWKU7GN?usp=sharing)

**4. Save và tải model:**

Tạm gọi mô hình đã được huấn luyện có tên là signal.pt

Tới bước này ta đã xong bước huấn luyện YOLO, ta sẽ triển khai YOLO để cắt được phần ảnh hợp lí nhất sau đó đưa vào CNN. Giả sử đã cắt được bước hình biển báo giao thông rồi và bây giờ cần phân loại, ta tiếp tục các bước sau đối với CNN

**Bước 6: [CNN] Lựa chọn model và data phù hợp:**
Tham khảo model + data biển báo giao thông được lấy tại [đây](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed) của tác giả này.

**Bước 6: [CNN] Data augmentation:**
Dựa vào data đã thu thập được, sau đó ta thu thập thêm data thực tế của mình và gán nhãn. Sau đó thực hiện một số kĩ thuật data augmentation tại [đây](https://www.tensorflow.org/tutorials/images/data_augmentation). Có thể ghép thêm nền của các vật cảnh khác từ nhữg hình có sẵn để tăng độ chính xác và tổng quan của model

**Bước 7: [CNN] Huấn luyện model CNN:**
Tham khảo tại file sau: [đây](https://github.com/CEK19/Autonomous-Car/blob/document/src/trafficSignDetection/train.ipynb)

**Tổng kết**: Toàn bộ quá trình từ việc triển khai model signal.pt để cắt hình hợp lí nhất, sau đó đưa vào CNN có thể tham khảo tại: [Ref YOLO + CNN](https://github.com/CEK19/Autonomous-Car/blob/document/kot3_pkg/scripts/trafficSignV2.py). Về ý tưởng chung được miêu tả như sau:
- Sử dụng YOLO để cắt hình biển báo ra khỏi hình tổng. Trên một hình có thể có nhiều biển báo và ta ưu tiên vị trí boundary box có độ chính xác lớn nhất mà thoả ngưỡng về kích cỡ.
- Hình biển báo sau khi được crop ra sẽ được đưa vào khối CNN để phân loại 1 trong 5 loại biển báo đã đề ra.
- Lưu ý rằng nên thay đổi kiến trúc model CNN để được độ chính xác cao hơn.


**Module nhận diện và phân loại đèn giao thông**
- Các cách tiếp cận:
- Hướng tiếp cận đang được sử dụng:
