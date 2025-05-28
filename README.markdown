# Credit Card Fraud Detection Using Machine Learning

## Overview
This project implements a **Credit Card Fraud Detection System** using machine learning techniques. It leverages the [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle to train and evaluate models for identifying fraudulent credit card transactions. The system uses Logistic Regression, Decision Tree, and Random Forest classifiers, with the Random Forest model achieving the highest accuracy after applying SMOTE to handle class imbalance.

## Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and contains 284,807 transactions with 31 features:
- **Time**: Seconds elapsed between each transaction and the first transaction.
- **V1 to V28**: Principal components obtained via PCA for anonymized features.
- **Amount**: Transaction amount.
- **Class**: Target variable (0 for non-fraudulent, 1 for fraudulent).

The dataset is highly imbalanced, with only 0.17% of transactions labeled as fraudulent (492 fraudulent vs. 284,315 non-fraudulent). To address this, **SMOTE** (Synthetic Minority Over-sampling Technique) is used to balance the classes, resulting in 275,190 samples for each class.

## Project Structure
- **Credit Card Fraud Detection Using Machine Learning.ipynb**: Jupyter Notebook containing the code for data exploration, preprocessing, model training, and evaluation.
- **creditcard.csv**: The dataset file (not included in the repository; download from Kaggle).
- **credit_card_model.pkl**: Saved Random Forest model (generated during execution, not included in the repository).
- **README.md**: This file, providing an overview and instructions for the project.

## Requirements
To run this project, you need the following Python libraries:
```bash
pandas
numpy
scikit-learn
imbalanced-learn
seaborn
matplotlib
```

Install the dependencies using:
```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project directory.
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "Credit Card Fraud Detection Using Machine Learning.ipynb"
   ```

## Usage
1. **Data Exploration**:
   - The notebook includes steps to load the dataset, display the first and last five rows, check the shape (284,807 rows, 31 columns), and verify for null values (none found).
   - Basic statistics and data types are explored using `data.info()`.

2. **Preprocessing**:
   - Features (`X`) are separated from the target variable (`y`, `Class`).
   - **SMOTE** is applied to address class imbalance, balancing the dataset to 275,190 samples per class.
   - The data is split into training (80%) and testing (20%) sets using `train_test_split`.

3. **Model Training and Evaluation**:
   - Three models are trained and evaluated on the balanced dataset:
     - **Logistic Regression**:
       - Accuracy: 94.51%
       - Precision: 97.27%
       - Recall: 91.58%
       - F1-Score: 94.34%
     - **Decision Tree**:
       - Accuracy: 99.84%
       - Precision: 99.78%
       - Recall: 99.89%
       - F1-Score: 99.84%
     - **Random Forest**:
       - Accuracy: 99.99%
       - Precision: 99.99%
       - Recall: 100%
       - F1-Score: 99.99%
   - A bar plot visualizes model accuracies using `seaborn`.

4. **Saving the Model**:
   - The Random Forest model, which performs best, can be saved using `joblib` as `credit_card_model.pkl` (not included in the repository but generated during execution).

## Model Performance
The performance of the models on the test set (after SMOTE) is summarized below:

| Model             | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------------------|--------------|---------------|------------|--------------|
| Logistic Regression | 94.51        | 97.27         | 91.58      | 94.34        |
| Decision Tree      | 99.84        | 99.78         | 99.89      | 99.84        |
| Random Forest      | 99.99        | 99.99         | 100.00     | 99.99        |

The Random Forest model is selected for its superior performance.

## Example Prediction
To test the model manually:
```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("credit_card_model.pkl")

# Example input (30 features: Time, V1-V28, Amount)
input_data = [[0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787, 
               0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791, 
               0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, 
               -0.021053, 149.62]]

# Predict
pred = model.predict(input_data)
print("Normal Transaction" if pred[0] == 0 else "Fraudulent Transaction")
```

## Notes
- The dataset features (V1 to V28) are PCA-transformed for anonymity, making them non-interpretable in their raw form.
- The notebook includes model training both before and after SMOTE. The results after SMOTE are used in the final evaluation due to better handling of class imbalance.
- The Random Forest model is recommended for deployment due to its near-perfect performance.
- The notebook does not include a GUI, but one can be added using Tkinter for real-time predictions (see Future Improvements).

## Future Improvements
- Implement hyperparameter tuning (e.g., using GridSearchCV) for the Random Forest model to optimize performance.
- Add cross-validation to ensure robust model evaluation.
- Develop a Tkinter-based GUI for user-friendly input and prediction.
- Explore advanced algorithms like XGBoost or neural networks for potential performance gains.
- Include feature importance analysis to understand key predictors of fraud.

## Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset provided by [MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Built using Python, scikit-learn, imbalanced-learn, pandas, numpy, seaborn, and matplotlib.