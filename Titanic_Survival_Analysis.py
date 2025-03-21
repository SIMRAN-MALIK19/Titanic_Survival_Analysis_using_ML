import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import schedules
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


import warnings
warnings.filterwarnings("ignore")
#importing data files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
nan_values = train.isna().sum()

# Display columns with NaN values in train.csv
columns_with_nan_values = nan_values[nan_values > 0].index
print("Columns with NaN values:")
print(columns_with_nan_values)

#ensuring reproducibility
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from sklearn.utils import check_random_state
random_state = check_random_state(42)

#preprocessing data by substituting NaN values, One Hot Encoding Categorical Variables and Normailising all numerical data
def input_preprocessing(input_data):
    processing_data = input_data.copy()
    #remove unimportant columns that will not be helpful for training/testing
    if 'Survived' in processing_data.columns:
        remove_cols = ['Name', 'Cabin', 'Ticket', 'PassengerId']
        processing_data.drop(columns=remove_cols, inplace=True)
    else:
        remove_cols = ['Name', 'Cabin', 'Ticket',] #retaining PassengerId for test.csv data as it will be used later for mapping pedictions 
        processing_data.drop(columns=remove_cols, inplace=True)
        processing_data['Fare'] = processing_data['Fare'].fillna(processing_data.groupby("Pclass")["Fare"].transform("mean"))
    #filling NaN values with appropriate values
    processing_data['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
    processing_data["Age"] = processing_data["Age"].fillna(processing_data.groupby("Sex")["Age"].transform("median"))

    processing_data['Embarked'] = processing_data['Embarked'].fillna(train['Embarked'].mode())
    processing_data = pd.get_dummies(processing_data, columns=['Embarked'], dtype=np.float64)

    #standard normalising all numerical variables
    columns_to_normalize = ["Age", "SibSp", "Parch", "Fare"] #standard normalising all numerical variables
    scaler = StandardScaler()
    processing_data[columns_to_normalize] = scaler.fit_transform(processing_data[columns_to_normalize])

    return processing_data

# Preprocess the training data
new_train = input_preprocessing(train)

X = new_train.drop(columns='Survived')
Y = new_train['Survived']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Support Vector Machine (SVM) with cross-validation
svc = svm.SVC(C=0.5, kernel='rbf', gamma='auto')

# Cross-validation for SVM
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_accuracies = []

for train_index, test_index in skf.split(X, Y):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[test_index]
    Y_train_cv, Y_val_cv = Y.iloc[train_index], Y.iloc[test_index]

    svc.fit(X_train_cv, Y_train_cv)
    accuracy = svc.score(X_val_cv, Y_val_cv)
    svm_accuracies.append(accuracy)

# Calculate and print the mean accuracy across all folds for SVM
mean_svm_accuracy = np.mean(svm_accuracies)
print("Mean Cross-Validation Accuracy for SVM:", mean_svm_accuracy)

# Neural Network with cross-validation
input_dim = X_train.shape[1]

# Create a fully connected neural network model
def create_nn_model():
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))  # Adjust dropout rate as needed
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))  # Adjust dropout rate as needed
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))  # Adjust dropout rate as needed
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))  # Adjust dropout rate as needed
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compile the model
nn_model = create_nn_model()
lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.001)
nn_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

# Cross-validation for Neural Network
nn_accuracies = []

for train_index, test_index in skf.split(X, Y):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[test_index]
    Y_train_cv, Y_val_cv = Y.iloc[train_index], Y.iloc[test_index]

    # Fit the model to the training data with early stopping
    nn_model.fit(X_train_cv, Y_train_cv, epochs=10, batch_size=64, validation_data=(X_val_cv, Y_val_cv),
                 callbacks=[EarlyStopping(patience=3)])

    # Evaluate the model on the validation data
    _, accuracy = nn_model.evaluate(X_val_cv, Y_val_cv)
    nn_accuracies.append(accuracy)

# Calculate and print the mean accuracy across all folds for Neural Network
mean_nn_accuracy = np.mean(nn_accuracies)
print("Mean Cross-Validation Accuracy for Neural Network:", mean_nn_accuracy)

# Random Forest Classifier with cross-validation
rf_accuracies = []

for train_index, test_index in skf.split(X, Y):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[test_index]
    Y_train_cv, Y_val_cv = Y.iloc[train_index], Y.iloc[test_index]

    rf_classifier = RandomForestClassifier(max_depth=3,n_estimators=100, ccp_alpha=0.001, max_features='sqrt', random_state=42)
    rf_classifier.fit(X_train_cv, Y_train_cv)
    accuracy = rf_classifier.score(X_val_cv, Y_val_cv)
    rf_accuracies.append(accuracy)

# Calculate and print the mean accuracy across all folds for Random Forest
mean_rf_accuracy = np.mean(rf_accuracies)
print("Mean Cross-Validation Accuracy for Random Forest:", mean_rf_accuracy)

#training all 3 models on full training split set and Testing on test split set
nn_model1 = create_nn_model()
lr_schedule1 = schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.001)
nn_model1.compile(optimizer=Adam(learning_rate=lr_schedule1), loss='binary_crossentropy', metrics=['accuracy'])
nn_model1.fit(X_train, Y_train, epochs=1000, batch_size=64, validation_split=0.2,callbacks=[EarlyStopping(patience=3)])
train_accuracy = nn_model1.evaluate(X_train, Y_train)[1]
print("Training Accuracy on complete training data using nn model:", train_accuracy)

accuracy = nn_model1.evaluate(X_test, Y_test)[1]
print("Test Accuracy (test split from train.csv) with nn model:", accuracy)

svc.fit(X_train_cv, Y_train_cv)
train_accuracy1 = svc.score(X_train, Y_train)
print("Training Accuracy on complete training data using svm model:", train_accuracy1)
print("Test Accuracy (test split from train.csv) with svm model:",svc.score(X_test,Y_test))

rf_classifier1 = RandomForestClassifier(max_depth=3,n_estimators=100, ccp_alpha=0.001,max_features='sqrt', random_state=42)
rf_classifier1.fit(X_train, Y_train)
train_accuracy2 = rf_classifier1.score(X_train, Y_train)
print("Training Accuracy on complete training data using RF model:", train_accuracy2)
accuracy1= rf_classifier1.score(X_test, Y_test)
print("Test Accuracy (test split from train.csv) with RF model:", accuracy1)
#evaluating and comparing performance metrics for all 3 models based on testing performance
#confusion matrix

# Confusion Matrix for SVM
svm_pred = svc.predict(X_test) ##
svm_cm = confusion_matrix(Y_test, svm_pred)

# Plot SVM Confusion Matrix
sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.show()

# Confusion Matrix for Neural Network
nn_pred_prob = nn_model1.predict(X_test)
nn_pred = (nn_pred_prob > 0.5).astype(int)
nn_cm = confusion_matrix(Y_test, nn_pred) ##

# Plot NN Confusion Matrix
sns.heatmap(nn_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Neural Network Confusion Matrix")
plt.show()

# Confusion Matrix for Random Forest
rf_pred = rf_classifier1.predict(X_test) ##
rf_cm = confusion_matrix(Y_test, rf_pred)

# Plot RF Confusion Matrix
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

#ROC CURVE

# ROC Curve for SVM
svm_fpr, svm_tpr, _ = roc_curve(Y_test, svm_pred)
svm_auc = auc(svm_fpr, svm_tpr)

# Plot SVM ROC Curve
plt.plot(svm_fpr, svm_tpr, label=f'SVM AUC = {svm_auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend()
plt.show()

# ROC Curve for Neural Network
nn_fpr, nn_tpr, _ = roc_curve(Y_test, nn_pred)
nn_auc = auc(nn_fpr, nn_tpr)

# Plot NN ROC Curve
plt.plot(nn_fpr, nn_tpr, label=f'NN AUC = {nn_auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network ROC Curve')
plt.legend()
plt.show()

# ROC Curve for Random Forest
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_pred)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot RF ROC Curve
plt.plot(rf_fpr, rf_tpr, label=f'RF AUC = {rf_auc}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve for SVM
svm_precision, svm_recall, _ = precision_recall_curve(Y_test, svm_pred)

# Plot SVM Precision-Recall Curve
plt.plot(svm_recall, svm_precision, label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('SVM Precision-Recall Curve')
plt.legend()
plt.show()

# Precision-Recall Curve for Neural Network
nn_precision, nn_recall, _ = precision_recall_curve(Y_test, nn_pred)

# Plot NN Precision-Recall Curve
plt.plot(nn_recall, nn_precision, label='Neural Network')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Neural Network Precision-Recall Curve')
plt.legend()
plt.show()

# Precision-Recall Curve for Random Forest
rf_precision, rf_recall, _ = precision_recall_curve(Y_test, rf_pred)

# Plot RF Precision-Recall Curve
plt.plot(rf_recall, rf_precision, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Random Forest Precision-Recall Curve')
plt.legend()
plt.show()


# Function to calculate and return metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Calculate metrics for SVM
svm_accuracy, svm_precision, svm_recall, svm_f1 = calculate_metrics(Y_test, svm_pred)

# Calculate metrics for Neural Network
nn_accuracy, nn_precision, nn_recall, nn_f1 = calculate_metrics(Y_test, nn_pred)

# Calculate metrics for Random Forest
rf_accuracy, rf_precision, rf_recall, rf_f1 = calculate_metrics(Y_test, rf_pred)

# Create a bar graph
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
svm_metrics = [svm_accuracy, svm_precision, svm_recall, svm_f1]
nn_metrics = [nn_accuracy, nn_precision, nn_recall, nn_f1]
rf_metrics = [rf_accuracy, rf_precision, rf_recall, rf_f1]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, svm_metrics, width, label='SVM')
rects2 = ax.bar(x, nn_metrics, width, label='Neural Network')
rects3 = ax.bar(x + width, rf_metrics, width, label='Random Forest')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()


# Display columns with NaN values in test.csv file
nan_values1 = test.isna().sum()
columns_with_nan_values1 = nan_values1[nan_values1 > 0].index
print("Columns with NaN values:")
print(columns_with_nan_values1)

#preprocessing data from test.csv
X_testing=input_preprocessing(test)
#storing PassengerID separately for preprocessed data from test.csv file
passenger_ids = X_testing['PassengerId']
#dropping passengerId for making predictions for 'Survived' attribute
X_testing.drop(columns=['PassengerId'], inplace=True)
test_nn_predictions = nn_model1.predict(X_testing)
test_nn_predictions=np.array(test_nn_predictions)

threshold_value = 0.5
test_nn_predictions = (test_nn_predictions > threshold_value).astype(int)
#mapping the predictions for 'Survived' attribute with corresponding PassengerId's and storing in Results.csv
Final_output_prediction = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_nn_predictions.flatten()})
Final_output_prediction.to_csv('Results.csv', index=False)