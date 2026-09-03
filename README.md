# AdaBoost Classification – Census Income Prediction

## 📌 Project Overview

This project demonstrates the use of the **AdaBoost (Adaptive Boosting) classification algorithm** to predict whether an individual's annual income is **greater than $50K or less than/equal to $50K**.

The project uses the Census Income dataset and covers the complete machine learning workflow, including:

* Data loading and exploration
* Missing-value detection and handling
* Duplicate-value removal
* Exploratory data analysis
* Categorical feature encoding
* Train-test splitting
* AdaBoost model training
* Model prediction
* Accuracy evaluation

## 📊 Dataset

The dataset contains **32,561 records and 15 columns** before duplicate removal.

### Features

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `age`            | Age of the individual                 |
| `workclass`      | Type of employment                    |
| `fnlwgt`         | Final sampling weight                 |
| `education`      | Education level                       |
| `education-num`  | Numerical representation of education |
| `marital-status` | Marital status                        |
| `occupation`     | Occupation                            |
| `relationship`   | Relationship status                   |
| `race`           | Race                                  |
| `sex`            | Gender                                |
| `capital-gain`   | Capital gains                         |
| `capital-loss`   | Capital losses                        |
| `hours-per-week` | Hours worked per week                 |
| `native-country` | Country of origin                     |
| `annual_income`  | Target variable                       |

The target variable is `annual_income`, with two classes:

* `<=50K`
* `>50K`

The notebook loads the dataset using Pandas and performs initial data inspection.

## 🧹 Data Preprocessing

### Missing Values

The dataset initially represents missing categorical values using `"?"`. These values are replaced with `NaN`.

Missing values were found in:

* `workclass`
* `occupation`
* `native-country`

The missing categorical values are then filled using the **mode** of the respective column.

### Duplicate Removal

The dataset contained **24 duplicate rows**, which were removed before model training. After removing duplicates, the dataset contained **32,537 records**.

### Exploratory Data Analysis

Box plots are generated for the numerical columns to inspect their distributions and identify potential outliers.

## 🔢 Feature Encoding

Categorical columns are converted into numerical values using Scikit-learn's **LabelEncoder**.

The notebook applies label encoding to every column whose data type is `object`.

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)
```

The `random_state=42` ensures reproducibility.

## 🤖 AdaBoost Model

The project uses Scikit-learn's:

```python
from sklearn.ensemble import AdaBoostClassifier
```

Several AdaBoost configurations are explored in the notebook. One of the final configurations uses:

```python
AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1,
    random_state=42
)
```

The model is trained on the training dataset and used to make predictions on the test dataset.

## 📈 Model Evaluation

The model predictions are evaluated using **accuracy score**:

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

Accuracy is calculated for the different models explored in the notebook.

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Google Colab / Jupyter Notebook**

## 📁 Project Structure

```text
.
├── Adaboost.ipynb
├── census_income.csv
└── README.md
```

> Make sure the dataset path in the notebook matches the location of `census_income.csv`.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Add the dataset

Place `census_income.csv` in the appropriate location used by the notebook.

### 4. Open the notebook

```bash
jupyter notebook Adaboost.ipynb
```

Alternatively, upload the notebook to **Google Colab**.

### 5. Run all cells

Execute the cells sequentially to perform preprocessing, train the AdaBoost models, generate predictions, and evaluate accuracy.

## 🎯 Objective

The main objective of this project is to understand how **AdaBoost classification** can be applied to a real-world classification problem and to follow the complete machine learning pipeline from raw data preprocessing to model evaluation.

## 📚 Key Concepts Demonstrated

* Data preprocessing
* Handling missing values
* Duplicate removal
* Exploratory data analysis
* Label encoding
* Train-test splitting
* Ensemble learning
* AdaBoost classification
* Model prediction
* Accuracy evaluation
