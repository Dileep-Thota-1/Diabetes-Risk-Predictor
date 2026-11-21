# Diabetes-Risk-Predictor

# 🧠 Pima Indians Diabetes Prediction

A complete machine learning pipeline for predicting the onset of diabetes using the Pima Indians dataset. This project includes preprocessing, model training, evaluation, class balancing with SMOTE, hyperparameter tuning, ensemble methods, and explainability using SHAP.

---

## 📊 Dataset

The dataset used is the **Pima Indians Diabetes Database**, containing medical diagnostic measurements for females aged 21 and older. The target variable `Outcome` indicates the presence (1) or absence (0) of diabetes.

- Features include: Glucose, BloodPressure, BMI, Insulin, SkinThickness, etc.
- Shape: `768 samples x 9 columns`

---

## 🧱 Project Structure

diabetes_prediction/  
│  
├── diabetes_model.pkl # Saved best model (Random Forest)  
├── scaler.pkl # Saved StandardScaler  
├── Pima_Indians_Diabetes_Database.csv # Input dataset  
├── diabetes_pipeline.ipynb # Main notebook with full pipeline  
└── README.md # Project documentation  


---

## 🛠️ Tech Stack

- **Python Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `shap`, `plotly`
- **ML Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Neural Network
- **Explainability**: SHAP for model interpretability

---

## 🔄 Workflow Overview

### 1. **Preprocessing**
- Replaces invalid 0s with NaN in `Glucose`, `BloodPressure`, etc.
- Applies `KNNImputer` to handle missing values.
- Uses `StandardScaler` for feature normalization.

### 2. **Exploratory Data Analysis**
- Count plots, correlation heatmap.

### 3. **Model Training & Evaluation**
- Trains 6 models on SMOTE-resampled data.
- Evaluates with Accuracy, AUC, Classification Report.
- Plots ROC curves for each model.

### 4. **Hyperparameter Tuning**
- Uses `GridSearchCV` on `RandomForestClassifier`.

### 5. **Ensemble Learning**
- Implements a `VotingClassifier` with Logistic Regression, Tuned Random Forest, and SVM.

### 6. **Model Explainability**
- Applies `SHAP` to visualize global and local feature importance.
- Uses modern SHAP API to explain predictions for class 1 (diabetes presence).

---

## ✅ Results

| Model               | Accuracy | AUC   |
|--------------------|----------|-------|
| Random Forest       |  **~86%**| **~0.91** |
| Logistic Regression |  ~82%    | ~0.86 |
| SVM                 |  ~83%    | ~0.88 |
| Gradient Boosting   |  ~85%    | ~0.89 |
| Voting Classifier   |  **~87%**| **~0.92** |

---

## 🧠 Explainability Insights

SHAP values provide feature impact at both **global** and **individual** levels:

- **Top influential features**: `Glucose`, `BMI`, `Age`
- Visualized using:
  - SHAP Beeswarm Plot
  - SHAP Force Plot

---

## 💾 Model Saving

The final tuned model and scaler are saved as:

```python
joblib.dump(best_model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
