# model_evaluation.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight

# For Recommendation System Evaluation
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy as surprise_accuracy

# Import the preprocessing function to get the base DataFrame
from data_preprocessing import preprocess_data

# --- Configuration ---
BASE_DIR = os.getcwd()
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, 'evaluation_results')
os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
print(f"Evaluation results directory '{EVALUATION_RESULTS_DIR}' ensured to exist.")

PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_data.pkl')

def evaluate_models():
    """
    Loads processed data, trained models, and preprocessors,
    then evaluates the performance of regression, classification,
    and recommendation models.
    """
    print("--- Starting Model Evaluation ---")

    # 1. Load Processed Data (to re-create splits for evaluation)
    try:
        df_main = preprocess_data(processed_output_folder=PROCESSED_DATA_DIR,
                                  processed_data_filename='processed_data')
        if df_main is None:
            print("Failed to load processed data. Exiting model evaluation.")
            return
        print(f"Processed data loaded for evaluation. Shape: {df_main.shape}")
    except Exception as e:
        print(f"Error loading processed data: {e}. Ensure data_preprocessing.py ran successfully.")
        return

    # Prepare data (same steps as in model_training.py before splitting)
    df_main.dropna(subset=['Rating'], inplace=True)
    df_main.dropna(subset=['VisitMode_Name'], inplace=True)

    id_cols_to_convert_int_actual = [
        'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId',
        'AttractionTypeId', 'CityId', 'VisitMode_FK'
    ]

    for col in id_cols_to_convert_int_actual:
        if col in df_main.columns:
            df_main[col] = pd.to_numeric(df_main[col], errors='coerce').astype('Int64')
            df_main[col] = df_main[col].fillna(-1).astype(int)
        
    # --- Load Models and Preprocessors ---
    print("\n--- Loading Trained Models and Preprocessors ---")

    try:
        # Load Regression Model and Scaler
        reg_model_path = os.path.join(MODELS_DIR, 'lightgbm_regressor.pkl')
        reg_scaler_path = os.path.join(MODELS_DIR, 'regression_scaler.pkl')
        reg_features_path = os.path.join(MODELS_DIR, 'regression_features.pkl')

        with open(reg_model_path, 'rb') as f:
            reg_model = pickle.load(f)
        with open(reg_scaler_path, 'rb') as f:
            scaler_reg = pickle.load(f)
        with open(reg_features_path, 'rb') as f:
            regression_features = pickle.load(f)
        print(f"  Regression model ({type(reg_model).__name__}), scaler, and features loaded.")

        # Load Classification Model and Label Encoder
        clf_model_path = os.path.join(MODELS_DIR, 'random_forest_classifier.pkl')
        clf_label_encoder_path = os.path.join(MODELS_DIR, 'classification_label_encoder.pkl')
        clf_features_path = os.path.join(MODELS_DIR, 'classification_features.pkl')

        with open(clf_model_path, 'rb') as f:
            clf_model = pickle.load(f)
        with open(clf_label_encoder_path, 'rb') as f:
            le = pickle.load(f)
        with open(clf_features_path, 'rb') as f:
            classification_features = pickle.load(f)
        print(f"  Classification model ({type(clf_model).__name__}), label encoder, and features loaded.")

        # Load Recommendation Model
        svd_model_path = os.path.join(MODELS_DIR, 'svd_recommendation_model.pkl')
        with open(svd_model_path, 'rb') as f:
            algo_svd = pickle.load(f)
        print(f"  Recommendation model ({type(algo_svd).__name__}) loaded.")

    except FileNotFoundError as e:
        print(f"ERROR: Model or preprocessor file not found: {e}. Ensure model_training.py was run successfully.")
        return
    except Exception as e:
        print(f"ERROR loading models/preprocessors: {e}.")
        return

    # --- Task 1: Evaluate Regression Model ---
    print("\n--- Evaluating Regression Model (Predicting Attraction Ratings) ---")

    # Re-create X_reg and y_reg for consistent splitting
    X_reg = df_main[regression_features].copy()
    y_reg = df_main['Rating']

    # Convert categorical features to 'category' dtype for LightGBM's inference (as done in training)
    categorical_features_for_lgbm_dt = [
        'VisitMonth', 'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId',
        'AttractionTypeId', 'CityId', 'VisitMode_FK'
    ]
    categorical_features_for_lgbm_dt_existing = [
        col for col in categorical_features_for_lgbm_dt if col in X_reg.columns
    ]
    for col in categorical_features_for_lgbm_dt_existing:
        X_reg[col] = X_reg[col].astype('category')

    # Apply scaler to numerical features in X_reg
    numerical_features_to_scale = ['VisitYear']
    numerical_features_to_scale_existing = [col for col in numerical_features_to_scale if col in X_reg.columns]
    if numerical_features_to_scale_existing:
        X_reg[numerical_features_to_scale_existing] = scaler_reg.transform(X_reg[numerical_features_to_scale_existing])

    # Split data to get the test set used for evaluation
    _, X_test_reg, _, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    y_pred_reg = reg_model.predict(X_test_reg)

    mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
    rmse_reg = np.sqrt(mse_reg)
    r2_reg = r2_score(y_test_reg, y_pred_reg)

    print(f"  Regression Model Performance:")
    print(f"    Mean Squared Error (MSE): {mse_reg:.4f}")
    print(f"    Root Mean Squared Error (RMSE): {rmse_reg:.4f}")
    print(f"    R-squared (R2): {r2_reg:.4f}")

    # --- Task 2: Evaluate Classification Model ---
    print("\n--- Evaluating Classification Model (Predicting User Visit Mode) ---")

    # Re-create X_clf and y_clf for consistent splitting
    X_clf = df_main[classification_features].copy()
    y_clf = df_main['VisitMode_Name']

    # Ensure integer/numerical types for features
    for col in X_clf.columns:
        X_clf[col] = pd.to_numeric(X_clf[col], errors='coerce').fillna(-1).astype(int)

    # Apply scaler to numerical features in X_clf (using the same scaler_reg)
    if numerical_features_to_scale_existing:
        X_clf[numerical_features_to_scale_existing] = scaler_reg.transform(X_clf[numerical_features_to_scale_existing])

    # Encode y_clf
    y_clf_encoded = le.transform(y_clf)

    # Split data to get the test set used for evaluation
    _, X_test_clf, _, y_test_clf = train_test_split(X_clf, y_clf_encoded, test_size=0.2, random_state=42, stratify=y_clf_encoded)

    y_pred_clf_encoded = clf_model.predict(X_test_clf)

    accuracy_clf = accuracy_score(y_test_clf, y_pred_clf_encoded)
    precision_clf = precision_score(y_test_clf, y_pred_clf_encoded, average='weighted')
    recall_clf = recall_score(y_test_clf, y_pred_clf_encoded, average='weighted')
    f1_clf = f1_score(y_test_clf, y_pred_clf_encoded, average='weighted')

    print(f"  Classification Model Performance:")
    print(f"    Accuracy: {accuracy_clf:.4f}")
    print(f"    Precision (weighted): {precision_clf:.4f}")
    print(f"    Recall (weighted): {recall_clf:.4f}")
    print(f"    F1-Score (weighted): {f1_clf:.4f}")

    # --- Task 3: Evaluate Recommendation System (scikit-surprise) ---
    print("\n--- Evaluating Recommendation System (scikit-surprise) ---")

    ratings_df_for_surprise = df_main[['UserId', 'AttractionId', 'Rating']].copy()
    min_rating = ratings_df_for_surprise['Rating'].min()
    max_rating = ratings_df_for_surprise['Rating'].max()
    reader = Reader(rating_scale=(min_rating, max_rating))
    data = Dataset.load_from_df(ratings_df_for_surprise, reader)

    _, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)

    predictions = algo_svd.test(testset)

    rmse_rec = surprise_accuracy.rmse(predictions, verbose=False)
    mae_rec = surprise_accuracy.mae(predictions, verbose=False)

    print(f"  Recommendation System (SVD) Performance:")
    print(f"    Root Mean Squared Error (RMSE): {rmse_rec:.4f}")
    print(f"    Mean Absolute Error (MAE): {mae_rec:.4f}")

    # --- Save Evaluation Results (Optional) ---
    print("\n--- Saving Evaluation Results ---")
    results_summary = {
        "Regression_MSE": mse_reg,
        "Regression_RMSE": rmse_rec,
        "Regression_R2": r2_reg,
        "Classification_Accuracy": accuracy_clf,
        "Classification_Precision_weighted": precision_clf,
        "Classification_Recall_weighted": recall_clf,
        "Classification_F1_weighted": f1_clf,
        "Recommendation_RMSE": rmse_rec,
        "Recommendation_MAE": mae_rec
    }

    results_file_path = os.path.join(EVALUATION_RESULTS_DIR, 'model_evaluation_summary.pkl')
    with open(results_file_path, 'wb') as f:
        pickle.dump(results_summary, f)
    print(f"  Evaluation summary saved to '{results_file_path}'.")

    with open(os.path.join(EVALUATION_RESULTS_DIR, 'model_evaluation_report.txt'), 'w') as f:
        f.write("--- Model Evaluation Report ---\n\n")
        f.write(f"Regression Model (LGBMRegressor):\n")
        f.write(f"  MSE: {mse_reg:.4f}\n")
        f.write(f"  RMSE: {rmse_rec:.4f}\n")
        f.write(f"  R2: {r2_reg:.4f}\n\n")
        f.write(f"Classification Model (RandomForestClassifier):\n")
        f.write(f"  Accuracy: {accuracy_clf:.4f}\n")
        f.write(f"  Precision (weighted): {precision_clf:.4f}\n")
        f.write(f"  Recall (weighted): {recall_clf:.4f}\n")
        f.write(f"  F1-Score (weighted): {f1_clf:.4f}\n\n")
        f.write(f"Recommendation System (SVD):\n")
        f.write(f"  RMSE: {rmse_rec:.4f}\n")
        f.write(f"  MAE: {mae_rec:.4f}\n")
    print(f"  Evaluation report saved to '{os.path.join(EVALUATION_RESULTS_DIR, 'model_evaluation_report.txt')}'.")


    print("\n--- Model Evaluation Complete ---")

if __name__ == "__main__":
    evaluate_models()
    print("--- End of model_evaluation.py execution ---")
