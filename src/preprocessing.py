import pandas as pd
import numpy as np
import warnings
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


warnings.filterwarnings('ignore')


PROCESSED_DATA_DIR = '../data/processed'
RAW_DATA = '../data/raw/healthifime_fitness_data.csv'
ARTIFACTS_DIR = '../artifacts'

def preprocess_data():
    try:
        df = pd.read_csv(RAW_DATA)
    except FileExistsError:
        print(f'File not found at location: {RAW_DATA}')

    # We saw some values from distolic and systolic BP columns were 0. Fixing that.
    df['diastolic'].replace(0, np.nan, inplace=True)
    df['systolic'].replace(0, np.nan, inplace=True)
    df['gripForce'].replace(0, np.nan, inplace=True)

    df.dropna(inplace=True)

    # Creating bmi and blood pressure features
    df['bmi'] = df['weight_kg']/((df['height_cm']/100)**2)
    df['blood_pressure'] = df['systolic']/df['diastolic']

    X = df.drop('class', axis=1)
    y = df['class']


    df['gender'] = df['gender'].astype('category').cat.codes

    label_target = LabelEncoder()
    y_encoded = label_target.fit_transform(y)
    joblib.dump(label_target, os.path.join(ARTIFACTS_DIR, 'target_encoder.joblib'))


    # Train test split
    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    X_train.shape, X_test.shape, X_val.shape
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))

    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_val_df = pd.DataFrame(X_val, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    y_train_df = pd.DataFrame(y_train, columns=['class'])
    y_val_df = pd.DataFrame(y_val, columns=['class'])
    y_test_df = pd.DataFrame(y_test, columns=['class'])

    X_train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    X_val_df.to_csv(os.path.join(PROCESSED_DATA_DIR,'X_val.csv'), index=False)
    X_test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    y_train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
    y_val_df.to_csv(os.path.join(PROCESSED_DATA_DIR,'y_val.csv'), index=False)
    y_test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)

    print(f"Preprocessing completed and files saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    preprocess_data()
