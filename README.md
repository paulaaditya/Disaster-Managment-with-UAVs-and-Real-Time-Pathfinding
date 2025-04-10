# Disaster-Managment-with-UAVs-and-Real-Time-Pathfinding

This project focuses on **autonomous navigation using aerial imagery**, combining **object detection** and **semantic segmentation** to generate road-aware paths while avoiding obstacles such as humans. The approach uses:

- **YOLO-based object detection** (on thermal UAV images)  
- **Semantic segmentation** (to classify roads from aerial RGB images)  
- **A\* pathfinding** (constrained to roads and avoiding obstacles)  
- **Patch-wise inference over a grid** for large-scale image processing

---

## 🔍 Objective

Design an autonomous routing system that:

- Detects **roads and backgrounds** using segmentation on high-resolution aerial images
- Detects **humans and obstacles** using object detection from drone-captured thermal images
- Uses this data to generate the **shortest safe route** to a target point, adhering to roads and avoiding obstacles using A\* algorithm

---

## 🧠 Key Features

- **Patch-wise Inference**: Divides aerial images into a 10x10 grid for scalable inference
- **YOLOv8**: Used for both segmentation and detection
- **A\* Algorithm**: Generates road-following paths, avoiding humans
- **Heatmap Overlay**: Highlights roads (from segmentation) and overlays people (from detection)
- **Custom Output Formatter**: Converts road segmentation masks into YOLO-compatible `.txt` labels

---

## 🗂 Datasets Used

### 🚀 **Road Segmentation**  
**Dataset**: [HD Maps Dataset](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/hd-maps)  
**Source**: German Aerospace Center (DLR)  
**Reference**:  
Mattyus, G., Wang, S., Fidler, S., Urtasun, R., 2016.  
"HD maps: Fine-grained road segmentation by parsing ground and aerial images."  
**Conference**: CVPR 2016  
🔗 [Paper Link](https://openaccess.thecvf.com/content_cvpr_2016/papers/Mattyus_HD_Maps_Fine-Grained_CVPR_2016_paper.pdf)

Used for **training road segmentation models**. The dataset provides aerial images with semantic color-coded masks (roads are `[255, 105, 180]`), which we convert to YOLO format.

---

### 🔥 **Object Detection**  
**Dataset**: [HIT-UAV Infrared-Thermal Dataset](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)  
**Description**: Infrared UAV dataset for human/object detection  
Used to train the YOLOv8 detection model to **identify people and obstacles** from thermal drone images.

---

## 🧰 Dependencies

Make sure your environment includes:

```txt
torch>=2.0
opencv-python
numpy
matplotlib
ultralytics==8.0.20
Pillow
```

You can install them using:

```bash
pip install -r requirements.txt
```

---

## 💻 Hardware Used

Training and inference were performed on the following hardware:

- **CPU**: Intel Core i9-14900HX 
- **GPU**: NVIDIA RTX 4070 8GB Mobile
- **RAM**: 32 GB DDR5 at 5600 MT/s

---

## 🚀 Usage

### 1. **Training**

run yolo-segmentation-train.ipynb

### 2. **Convert Road Masks to YOLO Format**

run convert-masks-to-yolo.py

> Make sure `road` class is index 0 in your segmentation mask and that colors are converted accurately.

### 3. **Pathfinding Pipeline**

run Disaster-Managment-with-UAVs-and-Real-Time-Pathfinding.ipynb

This runs segmentation and detection by dividing the image into a 10x10 grid for more accurate segmentation, merges results, and computes a path along roads while avoiding detected obstacles.

---

## 🗾 Citation & Acknowledgements

If you use this work or dataset, please cite:

```
@inproceedings{mattyus2016hd,
  title={HD maps: Fine-grained road segmentation by parsing ground and aerial images},
  author={Mattyus, Gellert and Wang, Shenlong and Fidler, Sanja and Urtasun, Raquel},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3611--3619},
  year={2016}
}
```

### External Repositories Referenced

- [HIT-UAV Infrared-Thermal Dataset](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)

---

## 📌 Notes

- The segmentation masks were preprocessed by identifying pinkish roads `(255, 105, 180)` and mapping them to YOLO format
- A\* pathfinder prioritizes roads and avoids obstacles, leveraging detection results
- If dataset license requires academic use only, ensure you adhere to that in all deployments

