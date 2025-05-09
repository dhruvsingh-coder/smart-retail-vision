# YOLOv8 Product Detection and Object Counter

##  Project Overview

This project uses YOLOv8 (You Only Look Once version 8) for custom object detection and counting on a retail or structured environment dataset. The trained model identifies three classes: `Product-Name`, `Empty-Space`, and `Other-Product`, and exports detection counts to a JSON file. Ideal for shelf analysis, inventory tracking, and visual automation.

---

##  Workflow Overview

```mermaid
flowchart TD
    A[Start] --> B[Define paths and data.yaml]
    B --> C[Train YOLOv8 model]
    C --> D[Load trained model]
    D --> E[Run prediction on test images]
    E --> F[Parse prediction results]
    F --> G[Count object classes]
    G --> H[Export counts to final_counts.json]
    H --> I[Download JSON (if in Colab)]
    I --> J[End]
```

---

##  Step-by-Step Explanation

### 1. **Training the Model**

* The training uses a custom dataset defined in a `data.yaml` file.
* Model used: `yolov8n.pt` (Nano version for faster training/testing)
* Training parameters: 30 epochs, image size 640, batch size 16.

```python
model.train(
    data=yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,
    name='custom_object_detection_model'
)
```

### 2. **Inference**

* After training, the best weights are loaded and used for predictions on the test dataset.
* Inference is done using:

```python
results = model.predict(test_images_path, save=False)
```

### 3. **Parsing Results and Counting**

* The script loops through the prediction results, extracts the class indices, maps them to human-readable names, and counts the occurrences.
* Uses `collections.Counter` to keep count.

```python
counts = Counter({name: 0 for name in class_names})
```

### 4. **Exporting JSON**

* The counts are saved to a `final_counts.json` file with readable indentation.

```python
with open(output_json_path, 'w') as f:
    json.dump(counts, f, indent=4)
```

### 5. **Colab Download (Optional)**

* If you're using Google Colab, the resulting JSON file can be downloaded directly:

```python
files.download(output_json_path)
```

---

## 🗂 Folder Structure

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

##  Classes Detected

1. **Product-Name**: Recognized items or known products on shelves.
2. **Empty-Space**: Gaps on the shelf indicating no product.
3. **Other-Product**: Misplaced or irrelevant products.

---

##  Requirements

```bash
pip install ultralytics
pip install matplotlib
pip install opencv-python
pip install google-colab  # if using Colab
```

---

##  Sample Output (`final_counts.json`)

```json
{
    "Product-Name": 123,
    "Empty-Space": 15,
    "Other-Product": 42
}
```

---

## 📈 Use Cases

* Retail shelf inventory analysis
* Automated planogram compliance
* Store stock-out detection
* Intelligent merchandising insights

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Future Improvements

* Integrate with a web dashboard for visualization
* Add support for real-time webcam input
* Improve model accuracy with larger datasets
* Deploy as a REST API using FastAPI or Flask

---
