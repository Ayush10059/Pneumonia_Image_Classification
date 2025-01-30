# **Pneumonia Image Classification 🏥🩺**  

## **Overview**  
This project utilizes **Deep Learning** techniques to classify chest X-ray images into **Pneumonia** and **Normal** categories. Pneumonia is a serious lung infection, and early detection using **automated medical imaging analysis** can significantly aid healthcare professionals in timely diagnosis and treatment.  

## **Dataset**  
- **Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Data Description:**  
  - Chest X-ray images labeled as **Normal** or **Pneumonia**  
  - Training, validation, and test sets  
  - Includes bacterial and viral pneumonia cases  

## **Deep Learning Model**  
### **1️⃣ Preprocessing & Data Augmentation**  
- **Normalization**: Scaling pixel values for better model convergence  
- **Data Augmentation**:  
  - Random rotation, zoom, and flipping to enhance model generalization  
  - Helps deal with limited medical datasets  

### **2️⃣ Model Architecture**  
Implemented **Convolutional Neural Networks (CNNs)** for image classification:  
- **Baseline Model:** Simple CNN with a few convolutional layers   

### **3️⃣ Training & Optimization**  
- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, Precision, Recall, F1-score, AUC-ROC  
- **Early Stopping & Learning Rate Scheduling** to prevent overfitting  

## **Results & Performance**  
- Achieved **high accuracy (>90%)** on test data  

## **Installation & Usage**  
### **🔹 Install Dependencies**  
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python
```

### **🔹 Run the Training Script**  
```bash
python train.py
```
OR, open and run the Jupyter Notebook:  
```bash
jupyter notebook Pneumonia_Classification.ipynb
```

### **🔹 Predict on New Images**  
```python
from model import load_model, predict_image
model = load_model("pneumonia_model.h5")
prediction = predict_image("chest_xray_sample.jpg", model)
print("Prediction:", prediction)
```

## **Model Deployment (Future Work)**  
- Convert model to **TF Lite** for mobile healthcare applications  
- Deploy as a **Flask API** for real-time X-ray diagnosis  

## **Key Takeaways**  
✅ Deep Learning can significantly improve pneumonia detection in X-ray images.   

## **Contributing**  
Open to contributions! Fork the repo, make improvements, and submit a pull request.  

## **License**  
This project is licensed under the **MIT License**.  
