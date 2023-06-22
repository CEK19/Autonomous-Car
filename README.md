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
**Guideline**:

**Module né vật cản**
- Các cách tiếp cận:
- Hướng tiếp cận đang được sử dụng:

**Module nhận diện làn đường**
- Các cách tiếp cận:
- Hướng tiếp cận đang được sử dụng:

**Module nhận diện và phân loại biển báo giao thông**
- Các cách tiếp cận: Có 2 cách tiếp cận chính gồm:
  - [1] OpenCV + CNN:
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
Tổng quan các bước gồm: Cấu hình file > Huấn luyện > Lưu model dưới định dạng "tenModel.pt"

- Config file
- Tạo folder
- Mẫu file huấn luyện
- 

**Module nhận diện và phân loại đèn giao thông**
- Các cách tiếp cận:
- Hướng tiếp cận đang được sử dụng:
