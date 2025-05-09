# **YOLOv8 PRODUCT DETECTION AND OBJECT COUNTER**

## **PROJECT OVERVIEW**

**This project utilizes YOLOv8 (You Only Look Once version 8) for training a custom object detection model. It identifies and counts three object classes — `Product-Name`, `Empty-Space`, and `Other-Product` — in retail shelf or structured environment images. It outputs a JSON file with the final counts.**

---

## **WORKFLOW OVERVIEW**

**1. Define paths and data.yaml**
**2. Train YOLOv8 model**
**3. Load trained model**
**4. Run inference on test images**
**5. Parse prediction results**
**6. Count object classes**
**7. Export counts to `final_counts.json`**
**8. Optionally download the result (for Colab)**

---

## **STEP-BY-STEP EXPLANATION**

### **1. TRAINING THE MODEL**

* **Dataset specified in `data.yaml`**
* **Base model: `yolov8n.pt` (YOLOv8 nano)**
* **Parameters: 30 epochs, 640 image size, batch size 16**

```python
model.train(
    data=yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,
    name='custom_object_detection_model'
)
```

### **2. INFERENCE**

* **Model uses best weights for inference**

```python
results = model.predict(test_images_path, save=False)
```

### **3. PARSING RESULTS & COUNTING**

* **Class indices mapped to names using `class_names` list**
* **`collections.Counter` used for tallying detections**

```python
counts = Counter({name: 0 for name in class_names})
```

### **4. EXPORTING TO JSON**

* **Counts saved to a JSON file with proper formatting**

```python
with open(output_json_path, 'w') as f:
    json.dump(counts, f, indent=4)
```

### **5. OPTIONAL: DOWNLOAD IN COLAB**

* **Directly download the JSON output**

```python
files.download(output_json_path)
```

---

## **FOLDER STRUCTURE**

```
project-root/
├── data.yaml
├── test/
│   └── images/
├── final_counts.json
├── script.py
├── runs/
│   └── detect/
│       └── custom_object_detection_model/
│           └── weights/
│               └── best.pt
```

---

## **CLASSES DETECTED**

* **Product-Name**: Recognized products on shelf
* **Empty-Space**: Empty or gap regions
* **Other-Product**: Irrelevant/misplaced products

---

## **REQUIREMENTS**

```bash
pip install ultralytics
pip install matplotlib
pip install opencv-python
pip install google-colab  # if using Colab
```

---

## **SAMPLE OUTPUT (`final_counts.json`)**

```json
{
    "Product-Name": 123,
    "Empty-Space": 15,
    "Other-Product": 42
}
```

---

## **USE CASES**

* **Retail shelf inventory analysis**
* **Automated planogram compliance**
* **Store stock-out detection**
* **Visual merchandising insights**

---

---

## **LICENSE**

**MIT License - see the [LICENSE](LICENSE) file for details.**

---

## **FUTURE IMPROVEMENTS**

* **Real-time webcam integration**
* **Deploy as REST API (FastAPI/Flask)**
* **Add dashboard visualizations**
* **Improve with more training data**

---
