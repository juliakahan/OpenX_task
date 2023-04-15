import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc, skm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
         "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
         "Horizontal_Distance_To_Fire_Points"] + [f"Wilderness_Area{i}" for i in range(1, 5)] + \
        [f"Soil_Type{i}" for i in range(1, 41)] + ["Cover_Type"]
df = pd.read_csv('/content/DATA/covtype.data', header=None, names= columns)

def preprocess(df):
    # Check for missing values
    missing_values = df.isna().sum()
    print(missing_values)

    # Fill missing values with the median
    df = df.fillna(df.median())
    missing_values = df.isna().sum()
    print(missing_values)

    return df


def heuristic(df):
    # Get the median of each column
    medians = df.median(axis = 0)

    # Classify the forest cover type based on the heuristic
    cover_type = []
    for i in range(len(df)):
        if df["Elevation"][i] > medians["Elevation"]:
            cover_type.append(1)
        elif df["Slope"][i] <= 5:
            cover_type.append(5)
        else:
            cover_type.append(2)
    cover_type.save('heuristic_model.h5')
    df["Cover_Type_Classification"] = cover_type
    return df["Cover_Type"], df["Cover_Type_Classification"]



def train_test_split_scale(df):
    X = df.drop("Cover_Type", axis = 1).values
    #changing labels to 0-6
    y = df["Cover_Type"].values - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def random_forest(df):
    X_train, X_test, y_train, y_test = train_test_split_scale(df)
    rf_clf = RandomForestClassifier(random_state = 42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    y_probs = rf_clf.predict_proba(X_test)

    # Subtract 1 from the labels to change them to 0-6
    y_test = y_test - 1
    y_pred = y_pred - 1
    rf_clf.save('rf_clf_model.h5')

    # Return the predicted values
    return  y_test, y_pred, y_probs, rf_clf


def logistic_regression(df):
    X_train, X_test, y_train, y_test = train_test_split_scale(df)
    lr_clf = LogisticRegression(random_state = 42, max_iter=1000)
    lr_clf.fit(X_train, y_train)
    y_pred = lr_clf.predict(X_test)
    y_probs = lr_clf.predict_proba(X_test)

    # Subtract 1 from the labels to change them to 0-6
    y_test = y_test - 1
    y_pred = y_pred - 1

    lr_clf.save('lr_clf_model.h5')

    # Return the predicted values
    return  y_test, y_pred, y_probs, lr_clf


def randomized_search():
    param_dist = {
        'num_hidden_layers': [1, 2, 3],
        'num_neurons': [32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.4],
    }

    # Define the randomized search
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)
    random_search = RandomizedSearchCV(model, param_dist, cv=3, n_iter=10, verbose=1)
    return random_search

def create_model(num_hidden_layers=1, num_neurons=64, dropout_rate=0.2):
    model = tf.keras.Sequential()
    model.add(layers.Dense(num_neurons, input_shape=(54,), activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    for i in range(num_hidden_layers):
        model.add(layers.Dense(num_neurons, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(7, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def neural_network(df):
    X_train, X_test, y_train, y_test = train_test_split_scale(df)
    random_search = randomized_search()
    random_search.fit(X_train, y_train)
    y_pred = random_search.predict(X_test)
    best_model = create_model(**random_search.best_params_)
    history = best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
    y_probs = best_model.predict_proba(X_test)
    # Subtract 1 from the labels to change them to 0-6
    y_test = y_test - 1
    y_pred = y_pred - 1
    best_model.save('nn_model.h5')

    # Return the predicted values
    return  y_test, y_pred, y_probs, best_model


def evaluate_model(y_test, y_pred, y_probs):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = skm.classification_report(y_test, y_pred, labels=range(1, 8), output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index = [i for i in range(1,8)], columns = [i for i in range(1,8)])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="g")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    return accuracy, f1, precision, recall, report

df = preprocess(df)

cover_type, cover_type_classification = heuristic(df)
y_test_rf, y_pred_rf, y_probs_rf, rf_clf = random_forest(df)
y_test_lr, y_pred_lr, y_probs_lr, lr_clf = logistic_regression(df)
y_test_nn, y_pred_nn, y_probs_nn, nn_clf = neural_network(df)

evaluate_model(y_test_rf, y_pred_rf, y_probs_rf)
evaluate_model(y_test_lr, y_pred_lr, y_probs_lr)
evaluate_model(y_test_nn, y_pred_nn, y_probs_nn)





