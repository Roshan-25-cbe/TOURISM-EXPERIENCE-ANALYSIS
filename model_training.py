# model_training.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight

# For Regression
from lightgbm import LGBMRegressor

# For Classification
from sklearn.ensemble import RandomForestClassifier

# For Recommendation (using scikit-surprise as decided)
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split # Used for internal Surprise evaluation if needed

# Import the preprocessing function to load the consolidated data
from data_preprocessing import preprocess_data

# --- Configuration ---
BASE_DIR = os.getcwd()
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Models output directory '{MODELS_DIR}' ensured to exist.")

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_data.pkl') # Ensure this matches output from data_preprocessing.py

def train_models():
    """
    Loads processed data, trains regression, classification, and recommendation models,
    and saves the trained models and associated preprocessors.
    Uses scikit-surprise for the recommendation system.
    """
    print("--- Starting Model Training ---")

    # 1. Load Processed Data
    try:
        # preprocess_data function already handles loading from pickle or reprocessing
        df_main = preprocess_data(processed_output_folder=PROCESSED_DATA_DIR,
                                  processed_data_filename='processed_data')
        if df_main is None:
            print("Failed to load processed data. Exiting model training.")
            return
        print(f"Processed data loaded for training. Shape: {df_main.shape}")
    except Exception as e:
        print(f"Error loading processed data: {e}. Ensure data_preprocessing.py ran successfully.")
        return

    # --- Initial Data Preparation (No One-Hot Encoding for main models) ---
    print("\n--- Preparing Data (without One-Hot Encoding for main models) ---")

    # Drop rows where 'Rating' is NaN if any slipped through, as it's our target for regression and recommendations
    df_main.dropna(subset=['Rating'], inplace=True)
    # Also drop rows where VisitMode_Name is NaN (if any) as it's a classification target
    df_main.dropna(subset=['VisitMode_Name'], inplace=True)

    # Convert ID columns to appropriate integer types (same as in data_preprocessing.py and app.py)
    id_cols_to_convert_int_actual = [
        'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId',
        'AttractionTypeId', 'CityId', # CityId is for Attraction's City
        'VisitMode_FK'
    ]

    for col in id_cols_to_convert_int_actual:
        if col in df_main.columns:
            # Convert to nullable integer first, then fillna and convert to int
            df_main[col] = pd.to_numeric(df_main[col], errors='coerce').astype('Int64')
            df_main[col] = df_main[col].fillna(-1).astype(int) # Fill NaNs with a placeholder like -1
            print(f"  Converted '{col}' to integer and handled NaNs.")
        else:
            print(f"  Warning: ID column '{col}' not found for type conversion. Please verify column names in data_preprocessing.py output.")


    # --- Task 1: Regression (Predicting Attraction Ratings) ---
    print("\n--- Task 1: Regression - Predicting Attraction Ratings ---")

    # Features for Regression Model (using numerical and integer ID columns as per document)
    regression_features = [
        'VisitYear', 'VisitMonth', # Visit details: Year, month
        'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId', # User demographics
        'AttractionTypeId', 'CityId', # Attraction attributes: Type, location (CityId)
        'VisitMode_FK' # Visit details: mode of visit
        # Note: "previous average ratings" is an engineered feature not included in current df_main.
    ]

    # Filter to actual columns present in df_main
    regression_features = [col for col in regression_features if col in df_main.columns and col != 'Rating']
    
    X_reg = df_main[regression_features].copy()
    y_reg = df_main['Rating']

    # CRITICAL FIX: Convert identified categorical ID columns to 'category' dtype
    # LightGBM's scikit-learn API can handle pandas CategoricalDtype directly.
    categorical_features_for_lgbm_dt = [
        'VisitMonth', # Treat month as categorical for LightGBM
        'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId',
        'AttractionTypeId', 'CityId',
        'VisitMode_FK'
    ]
    # Filter to existing columns that are intended to be categorical
    categorical_features_for_lgbm_dt_existing = [
        col for col in categorical_features_for_lgbm_dt if col in X_reg.columns
    ]

    for col in categorical_features_for_lgbm_dt_existing:
        X_reg[col] = X_reg[col].astype('category')
        print(f"  Converted '{col}' to 'category' dtype for LightGBM.")

    # Apply StandardScaler to continuous numerical features (if any, like VisitYear if it's large)
    numerical_features_to_scale = ['VisitYear'] # Example: if VisitYear ranges widely.
    numerical_features_to_scale_existing = [col for col in numerical_features_to_scale if col in X_reg.columns]

    scaler_reg = StandardScaler()
    if numerical_features_to_scale_existing:
        X_reg[numerical_features_to_scale_existing] = scaler_reg.fit_transform(X_reg[numerical_features_to_scale_existing])
        print(f"  Numerical features {numerical_features_to_scale_existing} scaled for regression task.")
    else:
        print("  No explicit numerical features found for scaling in regression task.")


    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    print(f"  Regression data split: Train {X_train_reg.shape}, Test {X_test_reg.shape}")

    # Save Regression Feature names and Scaler
    reg_features_path = os.path.join(MODELS_DIR, 'regression_features.pkl')
    with open(reg_features_path, 'wb') as f:
        pickle.dump(X_train_reg.columns.tolist(), f)
    print(f"  Regression feature names saved to '{reg_features_path}'.")

    reg_scaler_path = os.path.join(MODELS_DIR, 'regression_scaler.pkl')
    with open(reg_scaler_path, 'wb') as f:
        pickle.dump(scaler_reg, f)
    print(f"  Regression scaler saved to '{reg_scaler_path}'.")

    # Train Regression Model (LGBMRegressor)
    print("  Training LGBMRegressor for rating prediction...")
    reg_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    reg_model.fit(X_train_reg, y_train_reg)
    print("  LGBMRegressor trained.")

    reg_model_path = os.path.join(MODELS_DIR, 'lightgbm_regressor.pkl')
    with open(reg_model_path, 'wb') as f:
        pickle.dump(reg_model, f)
    print(f"  Regression model saved to '{reg_model_path}'.")

    print("\n" + "="*80 + "\n")


    # --- Task 2: Classification (Predicting User Visit Mode) ---
    print("\n--- Task 2: Classification - Predicting User Visit Mode ---")

    # Features for Classification Model (as per document)
    classification_features = [
        'VisitYear', 'VisitMonth',
        'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId',
        'AttractionTypeId'
    ]

    # Filter to actual columns in df_main
    classification_features = [col for col in classification_features if col in df_main.columns]

    X_clf = df_main[classification_features].copy()
    y_clf = df_main['VisitMode_Name']

    # Ensure integer/numerical types for features
    for col in X_clf.columns:
        X_clf[col] = pd.to_numeric(X_clf[col], errors='coerce').fillna(-1).astype(int)

    # Encode target variable (VisitMode_Name)
    le = LabelEncoder()
    y_clf_encoded = le.fit_transform(y_clf)
    print(f"  Visit Modes Encoded: {list(le.classes_)}")

    # Apply the same StandardScaler used for regression features
    if numerical_features_to_scale_existing:
        X_clf[numerical_features_to_scale_existing] = scaler_reg.transform(X_clf[numerical_features_to_scale_existing])
        print(f"  Numerical features {numerical_features_to_scale_existing} scaled for classification task using regression scaler.")
    else:
        print("  No explicit numerical features found for scaling in classification task.")

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf_encoded)
    print(f"  Classification data split: Train {X_train_clf.shape}, Test {X_test_clf.shape}")

    # Save Classification Feature names and LabelEncoder
    clf_features_path = os.path.join(MODELS_DIR, 'classification_features.pkl')
    with open(clf_features_path, 'wb') as f:
        pickle.dump(X_train_clf.columns.tolist(), f)
    print(f"  Classification feature names saved to '{clf_features_path}'.")

    clf_label_encoder_path = os.path.join(MODELS_DIR, 'classification_label_encoder.pkl')
    with open(clf_label_encoder_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"  Classification LabelEncoder saved to '{clf_label_encoder_path}'.")

    # Compute class weights for handling imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_clf),
        y=y_train_clf
    )
    class_weights_dict = dict(zip(np.unique(y_train_clf), class_weights))
    print(f"  Computed class weights for classification: {class_weights_dict}")

    sample_weights = pd.Series(y_train_clf).map(class_weights_dict)

    # Train Classification Model (RandomForestClassifier)
    print("  Training RandomForestClassifier for VisitMode prediction (with class weighting)...")
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_model.fit(X_train_clf, y_train_clf, sample_weight=sample_weights)
    print("  RandomForestClassifier trained.")

    clf_model_path = os.path.join(MODELS_DIR, 'random_forest_classifier.pkl')
    with open(clf_model_path, 'wb') as f:
        pickle.dump(clf_model, f)
    print(f"  Classification model saved to '{clf_model_path}'.")

    print("\n" + "="*80 + "\n")


    # --- Task 3: Recommendation (Personalized Attraction Suggestions - using scikit-surprise) ---
    print("\n--- Task 3: Recommendation - Personalized Attraction Suggestions (scikit-surprise) ---")

    print("  Preparing data for scikit-surprise recommendation system...")
    ratings_df_for_surprise = df_main[['UserId', 'AttractionId', 'Rating']].copy()

    # Define the rating scale from the actual data's min/max rating for the Reader
    min_rating = ratings_df_for_surprise['Rating'].min()
    max_rating = ratings_df_for_surprise['Rating'].max()
    reader = Reader(rating_scale=(min_rating, max_rating))

    # Load data into Surprise's Dataset format
    data = Dataset.load_from_df(ratings_df_for_surprise, reader)

    # Build the full training set.
    trainset = data.build_full_trainset()
    print(f"  Surprise trainset size: {trainset.n_ratings} ratings.")

    print("  Training SVD model for recommendation...")
    algo_svd = SVD(random_state=42)
    algo_svd.fit(trainset)
    print("  SVD model trained.")

    svd_model_path = os.path.join(MODELS_DIR, 'svd_recommendation_model.pkl')
    with open(svd_model_path, 'wb') as f:
        pickle.dump(algo_svd, f)
    print(f"  Recommendation model (SVD) saved to '{svd_model_path}'.")

    print("\n--- All Models Trained and Saved Successfully ---")
    print(f"Trained models and preprocessors are saved in the '{MODELS_DIR}' directory.")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    train_models()
    print("--- End of model_training.py execution ---")