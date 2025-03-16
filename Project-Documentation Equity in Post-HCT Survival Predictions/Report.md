# **Documentation: Equity in Post-HCT Survival Predictions**

This document provides a detailed explanation of the approach, methodology, and implementation of the solution for the **CIBMTR - Equity in Post-HCT Survival Predictions** competition. The goal of the project was to predict post-hematopoietic cell transplantation (HCT) survival outcomes while ensuring equity across different demographic groups. Below, we break down the process into clear sections to help readers understand what was done, how it was done, and the tools and techniques used.

---

## **1. Project Overview**

### **Objective**
The primary objective of this project was to build a robust machine learning model to predict post-HCT survival outcomes. The competition emphasized ensuring equity in predictions across different demographic groups, such as race, to avoid biased outcomes.

### **Key Challenges**
- Handling missing data in the dataset.
- Managing outliers in numerical features.
- Encoding categorical variables effectively.
- Ensuring model predictions are equitable across demographic groups.
- Combining multiple models to improve prediction accuracy.

### **Solution Approach**
The solution involved:
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and capping outliers.
2. **Feature Engineering**: Transforming the target variable using the Nelson-Aalen estimator and logit transformation.
3. **Model Training**: Using ensemble learning with XGBoost, LightGBM, and CatBoost models.
4. **Model Stacking**: Combining predictions from the base models using a neural network meta-model.
5. **Inference**: Generating predictions on the test dataset using the trained models.

---

## **2. Data Preprocessing**

### **2.1 Loading Data**
The training and test datasets were loaded using `pandas`. The training dataset contained features related to patient demographics, clinical data, and survival outcomes.

```python
import pandas as pd
train = pd.read_csv("/kaggle/input/cibmtr-data/train.csv")
```

### **2.2 Handling Missing Values**
Missing values were handled separately for numerical and categorical columns:
- **Numerical Columns**: Missing values were imputed using the **KNN Imputer** with `n_neighbors=7`.
- **Categorical Columns**: Missing values were imputed using the **Simple Imputer** with the `most_frequent` strategy.

```python
from sklearn.impute import KNNImputer, SimpleImputer

num_imputer = KNNImputer(n_neighbors=7)
train[num_cols] = num_imputer.fit_transform(train[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
```

### **2.3 Outlier Handling**
Outliers in numerical columns were capped using the **Interquartile Range (IQR)** method. Values below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR` were replaced with the respective bounds.

```python
def cap_outliers_auto(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

train = cap_outliers_auto(train, num_cols)
```

### **2.4 Feature Selection**
Unnecessary columns like `ID`, `efs`, `efs_time`, and `y` were removed. The remaining columns were used as features for model training.

```python
RMV = ["ID", "efs", "efs_time", "y"]
FEATURES = [c for c in train.columns if c not in RMV]
```

### **2.5 Encoding Categorical Variables**
Categorical variables were encoded using **label encoding** to convert them into numerical values.

```python
for col in train.select_dtypes(include=['object', 'category']).columns:
    train[col] = train[col].astype('category').cat.codes
```

---

## **3. Feature Engineering**

### **3.1 Nelson-Aalen Target Transformation**
The target variable (`efs_time`) was transformed using the **Nelson-Aalen estimator** to estimate the cumulative hazard function. This transformation helps in capturing the survival dynamics more effectively.

```python
from lifelines import NelsonAalenFitter

def create_nelson(data):
    naf = NelsonAalenFitter(nelson_aalen_smoothing=0)
    naf.fit(durations=data['efs_time'], event_observed=data['efs'])
    return naf.cumulative_hazard_at_times(data['efs_time']).values * -1

train["y_nel"] = create_nelson(train)
```

### **3.2 Logit Transformation**
The transformed target variable was further processed using a **logit transformation** to normalize its distribution.

```python
def logit_transform(y, eps=2e-2, eps_mul=1.1):
    y = (y - y.min() + eps) / (y.max() - y.min() + eps_mul * eps)
    return np.log(y / (1 - y))

train["y_transformed"] = logit_transform(train["y_nel"])
```

---

## **4. Model Training**

### **4.1 Cross-Validation**
A **Stratified K-Fold** cross-validation strategy was used to ensure equitable representation of different demographic groups (e.g., `race_group`) across folds.

```python
from sklearn.model_selection import StratifiedKFold

FOLDS = 20
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train, train["race_group"])):
    train.loc[val_idx, "fold"] = fold
```

### **4.2 Base Models**
Three base models were trained:
1. **XGBoost**: A gradient boosting model with `n_estimators=1000` and `max_depth=4`.
2. **LightGBM**: A gradient boosting model with `n_estimators=1000` and `max_depth=6`.
3. **CatBoost**: A gradient boosting model with `iterations=1000` and `depth=6`.

```python
# XGBoost
model_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, subsample=0.8)
model_xgb.fit(x_train, y_train)

# LightGBM
model_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, num_leaves=31)
model_lgb.fit(x_train, y_train)

# CatBoost
model_cat = cb.CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=6)
model_cat.fit(x_train, y_train)
```

### **4.3 Model Stacking**
The predictions from the base models were combined using a **neural network meta-model**. The meta-model consisted of:
- Three dense layers with batch normalization and dropout for regularization.
- A final linear output layer.

```python
meta_model = keras.Sequential([
    layers.Dense(256, kernel_initializer='he_normal', input_shape=(3,)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dropout(0.3),
    layers.Dense(128, kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dropout(0.3),
    layers.Dense(64, kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
])

meta_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
meta_model.fit(stacked_train, train["y_transformed"], epochs=30, batch_size=16)
```

---

## **5. Inference**

### **5.1 Loading Preprocessors and Models**
The preprocessors (imputers) and trained models were loaded and applied for inference.

```python
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")

test[num_cols] = num_imputer.transform(test[num_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])

xgb_models = joblib.load("xgboost_models.pkl")
lgb_models = joblib.load("lightgbm_models.pkl")
cat_models = joblib.load("catboost_models.pkl")

xgb_preds, lgb_preds, cat_preds = np.zeros(len(test)), np.zeros(len(test)), np.zeros(len(test))

for model in xgb_models:
    xgb_preds += model.predict(test[FEATURES]) / len(xgb_models)

for model in lgb_models:
    lgb_preds += model.predict(test[FEATURES]) / len(lgb_models)

for model in cat_models:
    cat_preds += model.predict(test[FEATURES]) / len(cat_models)
```

### **5.2 Making Predictions**
Predictions were generated using the base models and combined using the meta-model.

```python

meta_model = tf.keras.models.load_model("meta_model.h5")

stacked_test = np.vstack((xgb_preds, lgb_preds, cat_preds)).T
final_preds = meta_model.predict(stacked_test).flatten()
```

### **5.3 Saving Predictions**
The final predictions were saved to a CSV file for submission.

```python
submission = pd.DataFrame({"ID": test["ID"], "prediction": final_preds})
submission.to_csv("submission.csv", index=False)
```

---

## **6. Tools and Libraries Used**
- **Python Libraries**: `numpy`, `pandas`, `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `lifelines`, `tensorflow`, `keras`.
- **Preprocessing**: `KNNImputer`, `SimpleImputer`.
- **Model Training**: `XGBRegressor`, `LGBMRegressor`, `CatBoostRegressor`.
- **Model Stacking**: TensorFlow/Keras neural network.

---

## **7. Conclusion**  
This project showcased a comprehensive approach to predicting post-HCT survival outcomes while ensuring fairness across demographic groups. By leveraging multiple models and advanced techniques such as the Nelson-Aalen estimator and model stacking, the solution achieved robust and equitable predictions.  

Throughout the competition, **I made a total of 45 submissions**, continuously refining the model for better accuracy. After the competition, my best result secured **1729th position** with a **final score of 0.68872**.  
