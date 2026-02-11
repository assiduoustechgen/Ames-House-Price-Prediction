
### House Price Prediction 

This repository contains a machine learning project that predicts residential home prices using the famous [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from Kaggle. The project explores data preprocessing, feature engineering, and compares the performance of linear models against a deep neural network.

# Project Overview

The goal of this project is to predict the final price of each home (`SalePrice`) based on 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa.

The analysis is performed in a Jupyter Notebook and covers:

1. Data Cleaning: Handling missing values (imputation) and dropping irrelevant features.
2. Feature Engineering: Mapping ordinal variables, binary encoding, and One-Hot Encoding for categorical data.
3. Model Building: Implementing Ridge Regression, Linear Regression, and a Deep Neural Network (MLP).
4. Evaluation: Comparing models using RMSE (Root Mean Squared Error) and R² scores.

# 🛠️ Technologies Used

* Python 3.x
* Pandas & NumPy: Data manipulation and numerical operations.
* Matplotlib: Data visualization.
* Scikit-Learn:
* Preprocessing (`OneHotEncoder`, `PolynomialFeatures`)
* Models (`LinearRegression`, `Ridge`)
* Metrics (`mean_squared_error`, `r2_score`)
* TensorFlow / Keras: Building and training the Artificial Neural Network.

## Approach & Methodology

### 1. Data Preprocessing

* **Dropped Columns**: Removed features with excessive missing data or low relevance (`PoolQC`, `MiscFeature`, `Alley`).
* **Imputation**: Filled missing values for `LotFrontage` (mean), `GarageYrBlt` (mode), and `MasVnrArea` (mean).
* **Encoding**:
* Mapped specific columns like `Alley` and `Fence` to binary indicators.
* Applied `OneHotEncoder` to transform categorical variables into a machine-readable format.



### 2. Model Architectures

#### Classical Machine Learning

* Linear Regression: Used as a baseline model.
* Ridge Regression: Applied with an alpha of `0.2` to handle multicollinearity and prevent overfitting.
* Polynomial Features: Experimented with degree-1 polynomial transformations.

#### Deep Learning (TensorFlow/Keras)

A Sequential Multi-Layer Perceptron (MLP) was built with the following architecture:

* **Input Layer**: Matches the shape of processed features.
* **Hidden Layers**: Dense layers with 256, 128, 64, and 32 neurons using `ReLU` activation.
* **Regularization**:
* **L1 Regularization** (`0.008`) applied to kernels to enforce sparsity.
* **Dropout Layers** (0.2 - 0.3) to prevent overfitting during training.


* **Optimizer**: Adam (learning rate `0.00081`).

## 📈 Results

The models were evaluated on a validation set (20% split).

* **Linear/Ridge Regression**: Achieved an **R² score of ~0.87**.
* **Neural Network**: The model converged over 100 epochs, achieving a validation RMSE comparable to the linear baselines, demonstrating the ability to capture non-linear relationships in the data.

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/house-price-prediction.git

```


2. **Install dependencies:**
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow

```


3. **Run the Notebook:**
Open `House.ipynb` in Jupyter Notebook or Google Colab to view the analysis and training process.

## 🔮 Future Improvements

* Perform more extensive Hyperparameter Tuning (GridSearchCV) for the Ridge and Lasso models.
* Experiment with ensemble methods like Random Forest or XGBoost.
* Conduct deeper Exploratory Data Analysis (EDA) to visualize correlations between features.

---

