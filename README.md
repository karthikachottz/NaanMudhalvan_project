# NaanMudhalvan_project
About the Project 
This Python program demonstrates how to build a fraud detection model 
using the Random Forest Classifier. The code imports necessary libraries, 
loads the dataset, preprocesses the data, handles imbalanced data using 
Synthetic Minority Over-sampling Technique (SMOTE), trains a Random 
Forest Classifier model, and evaluates its performance.
Built With 
Python
pandas
scikit-learn
imbalanced-learn (imblearn)
Getting Started 
Prerequisites 
To run this code, you need to have Python installed on your system.
Installation 
Clone the repository or download the fraud_detection.py file to your local 
machine.
Install the required Python libraries using pip:
 pip install pandas scikit-learn imbalanced-learn 
Usage 
1. Download the dataset from the source mentioned below and save it as 
`fraudTest.csv` in the same directory as the `fraud_detection.py` file.
2. Open a terminal or command prompt and navigate to the directory 
containing `fraud_detection.py`.
3. Run the script using the following command:
 python fraud_detection.py
The script will load the dataset, preprocess it, train the Random Forest 
Classifier model, and evaluate its performance.
Dataset Source 
The dataset used in this project is named `fraudTest.csv`. You can obtain it 
from https://www.kaggle.com/datasets/kartik2112/fraud-detection
Code Description: 
The provided Python code is designed for building a fraud detection model 
using the Random Forest Classifier. It encompasses various data 
preprocessing steps, feature engineering, imbalanced data handling, model 
training, and model evaluation. The code can be summarized as follows:
1. Library Imports: The code begins by importing necessary Python libraries, 
including pandas for data manipulation, scikit-learn for machine learning, 
imbalanced-learn (SMOTE), and related modules for data preprocessing and 
model evaluation.
2. Data Loading and Exploration: It loads the dataset from a file named 
"fraudTest.csv" using the pandas library. After loading, it prints the first few 
rows of the dataset, provides basic information about the dataset, checks for 
missing values, displays summary statistics, and shows the distribution of 
fraudulent and non-fraudulent transactions.
3. Data Preprocessing: This section involves several steps:
 - Removing rows with missing values in the 'is_fraud' column.
 - Separating the dataset into feature variables (X) and the target variable 
(y).
 - Excluding non-numeric columns and performing one-hot encoding on 
categorical columns ('merchant' and 'category').
 - Handling missing values in the 'amt' column and scaling it using 
StandardScaler.
 - Removing duplicate rows and selecting a subset of columns for feature 
selection.
4. Handling Imbalanced Data with SMOTE: To address the issue of 
imbalanced data, the code uses Synthetic Minority Over-sampling Technique
(SMOTE) with a specified sampling strategy. SMOTE generates synthetic 
samples of the minority class (fraudulent transactions) to balance the 
dataset.
5. Data Splitting: The preprocessed data is divided into training and 
testing sets with a specified test size and a random seed for reproducibility.
6. Model Training: The code trains a machine learning model using the 
Random Forest Classifier. It configures the model with 100 decision trees 
(n_estimators) and a fixed random seed for consistency.
7. Making Predictions: The trained model is used to make predictions on 
the test set, classifying transactions as either fraudulent or non-fraudulent.
8. Model Evaluation: The code calculates and prints the accuracy of the 
model's predictions and provides a classification report, including metrics 
like precision, recall, F1-score, and support for both classes. This allows for 
an assessment of the model's performance in detecting fraudulent 
transactions.
Importance: 
Fraud detection is of paramount importance in various industries, including 
banking, e-commerce, and healthcare. This code provides a foundational 
framework for building a fraud detection system. The Random Forest 
Classifier is a popular choice for such applications due to its ability to handle 
complex data and imbalanced datasets. Effective fraud detection can save 
organizations substantial financial losses and maintain trust with customers.
Specification: 
- The code expects a dataset named "fraudTest.csv" to be present in the 
same directory.
- It is written in Python and relies on several libraries, including pandas, 
scikit-learn, and imbalanced-learn.
- The Random Forest Classifier is employed as the machine learning model, 
with 100 decision trees (n_estimators) and a fixed random seed 
(random_state).
- The code is specifically designed for binary classification, distinguishing 
between fraudulent and non-fraudulent transactions.
- It offers an accuracy score and a detailed classification report to assess the 
model's effectiveness in detecting fraudulent transactions.
This code serves as a starting point for developing more sophisticated fraud 
detection systems, which can be customized and extended based on 
specific requirements and datasets.
How to run the program: 
1. Install Required Libraries:
 Ensure that you have the necessary Python libraries installed. You can 
install them using pip if you haven't already:
 pip install pandas scikit-learn imbalanced-learn 
 2. Dataset Preparation:
 Download the dataset (fraudTest.csv) from the source mentioned in the 
code, and save it in the same directory as the Python script.
3. Run the Program:
 Open a terminal or command prompt and navigate to the directory 
containing the Python script (where you saved "fraud_detection.py" or any 
other name you may have given to the script).
4. Execute the script using the following command:
 python fraud_detection.py 
 5. The script will load the dataset, preprocess the data, train the Random 
Forest Classifier model, make predictions, and display the accuracy and a 
classification report.
6. The results of the model's performance, including accuracy and detailed 
classification metrics, will be printed in the terminal.
