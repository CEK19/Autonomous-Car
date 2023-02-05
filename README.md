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