# Titanic_Survival_Analysis_using_ML
The sinking of the Titanic stands as one of the most tragic disasters in history.
Regrettably, the insufficient number of lifeboats led to the loss of many lives. Survival on that fateful night seemed to involve an element of chance, yet certain factors may have played a role in getting rescued.

This project leverages **Machine Learning (ML)** models—**Support Vector Machine (SVM), Neural Network (NN), and Random Forest (RF)**—to predict passenger survival based on features like age, gender, class, and other travel-related factors.

Before training, preprocessing of data is done - null values substituted, one-hot encoding of categorical variables and normalization of all numerical data. Model Performances are then compared on the following performance metrics:
* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curves

###Dependencies
The following dependencies should be installed in the system before running the code. All required dependencies can be installed by running the following command:
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn

###Dataset
Both train.csv and test.csv should be saved in the working directory of the project.

###Running the project:
The project can be run by running the following command in terminal/ide (edit according to the python version that is available on your system):
python3 Titanic_Survival_Analysis.py

Running this code will preprocess the data, train and validate the models, generate performance metrics and plots for each model, and save the predictions in results.csv file.

For further insights, checkout the **Titanic Survival Analysis Report**.
