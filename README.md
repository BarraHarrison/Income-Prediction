# 💰 Income Prediction Model

## 📌 Introduction  
This project is a **machine learning model** built to predict whether an individual's income exceeds $50K per year based on census data. Using **pandas**, **seaborn**, and **scikit-learn**, I preprocessed the data, visualized correlations, and trained a **Random Forest Classifier** with hyperparameter tuning using **GridSearchCV**. 🚀  

---

## 📺 Using Pandas Concatenate  
To handle categorical variables, I used **pandas' `concat()`** function to apply **one-hot encoding** with `pd.get_dummies()`. This ensures categorical variables like `marital-status`, `relationship`, `race`, and `native-country` are transformed into numerical features for the model.

```python
data_frame = pd.concat([data_frame.drop("marital-status", axis=1),
                        pd.get_dummies(data_frame["marital-status"]).add_prefix("marital-status_")], axis=1)

data_frame = pd.concat([data_frame.drop("relationship", axis=1),
                        pd.get_dummies(data_frame.relationship).add_prefix("relationship_")], axis=1)
```
This allows the model to **understand categorical data** in a numerical format. ✅  

---

## 🏏 Using Lambda Expressions  
Lambda functions help **convert categorical labels into binary values** efficiently.  

- `gender`: Mapped to `1` if **Male**, `0` if **Female**.
- `income`: Mapped to `1` if **above 50K**, `0` if **below 50K**.

```python
data_frame["gender"] = data_frame["gender"].apply(lambda x: 1 if x == "Male" else 0)
data_frame["income"] = data_frame["income"].apply(lambda x: 1 if x == ">50K" else 0)
```
This simplifies **feature engineering** and prepares the dataset for machine learning. ⚡  

---

## 🎨 Visualizing with Seaborn Colour-Map  
I used **Seaborn's heatmap** to visualize feature correlations, helping us understand relationships in the data.  

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 12))
sns.heatmap(data_frame.corr(), annot=False, cmap="coolwarm")
```
The **coolwarm color map** makes it easier to spot highly correlated features, which can **help in feature selection**. 📊  

---

## 🔀 Splitting Data using sklearn train_test_split  
To evaluate our model properly, I split the dataset into **training** (80%) and **testing** (20%) sets using `train_test_split()`.

```python
from sklearn.model_selection import train_test_split

data_frame = data_frame.drop("fnlwgt", axis=1)  # Dropping an unnecessary column

train_data_frame, test_data_frame = train_test_split(data_frame, test_size=0.2)
```
This ensures that the model **does not overfit** and can generalize well to unseen data. ✅  

---

## 🌳 Training with RandomForestClassifier  
I trained a **Random Forest Classifier**, an ensemble model that improves accuracy by combining multiple decision trees.

```python
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(train_x, train_y)
```
This classifier is **robust**, reduces overfitting, and is great for handling **high-dimensional data**. 🚀  

---

## 🔍 Hyperparameter Tuning with GridSearchCV  
To optimize our model, I carried out **hyperparameter tuning** using `GridSearchCV`, testing different combinations of parameters.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 250],       # Number of trees in the forest
    "max_depth": [5, 10, 30, None],       # Depth of the trees
    "min_samples_split": [2, 4],          # Minimum samples to split a node
    "max_features": ["sqrt", "log2"]      # Maximum features considered for splitting
}

grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, verbose=10)
grid_search.best_estimator_
```
This ensures that I found the **best combination of hyperparameters** for improved accuracy and performance. 🎯  

---

## 🏆 Conclusion  
This project demonstrates how to **prepare, visualize, train, and optimize** a machine learning model for income prediction. I used:  
✅ **Pandas** for data preprocessing  
✅ **Lambda functions** for feature engineering  
✅ **Seaborn** for data visualization  
✅ **Scikit-learn** for model training and evaluation  

By fine-tuning the **Random Forest Classifier** with `GridSearchCV`, I achieved an **optimized and efficient prediction model**! 🚀  

### ⭐ Future Improvements  
- Experiment with other models like **Gradient Boosting** or **XGBoost** for comparison.  
- Try **feature selection** to remove redundant features and improve efficiency.  
- Deploy the model using **Flask or FastAPI** to create an interactive web app!  