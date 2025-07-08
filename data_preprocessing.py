# data_preprocessing.py
import pandas as pd
import os
import pickle
from data_loading import load_all_data # Import the function to load raw CSVs

def preprocess_data(data_folder='Data', raw_pickled_data_path='Pickled_Data/raw_datasets.pkl',
                    processed_output_folder='processed_data', processed_data_filename='processed_data'):
    """
    Loads raw data (from pickle or CSVs), performs cleaning and merging, and saves the consolidated DataFrame.
    If the processed pickle file exists, it loads that instead to save processing time.

    Args:
        data_folder (str): The name of the folder containing the raw CSV files.
        raw_pickled_data_path (str): The file path to load raw datasets from if they were pickled.
        processed_output_folder (str): The folder where the processed data files will be saved.
        processed_data_filename (str): The base filename for the processed data (e.g., 'processed_data' for .pkl and .csv).

    Returns:
        pd.DataFrame: The cleaned and consolidated DataFrame.
    """
    # Define full paths for processed files
    processed_pkl_path = os.path.join(processed_output_folder, f"{processed_data_filename}.pkl")
    processed_csv_path = os.path.join(processed_output_folder, f"{processed_data_filename}.csv")

    # Try loading the already processed data first
    if os.path.exists(processed_pkl_path):
        print(f"Loading processed data from {processed_pkl_path}...")
        try:
            with open(processed_pkl_path, 'rb') as f:
                df_main = pickle.load(f)
            print("Processed data loaded successfully.")
            return df_main
        except Exception as e:
            print(f"Error loading processed data from {processed_pkl_path}: {e}. Re-processing...")
    
    print("Processed data not found or failed to load. Starting raw data loading and preprocessing...")
    
    # Try loading raw data from pickle first
    datasets = None
    if os.path.exists(raw_pickled_data_path):
        print(f"Loading raw datasets from {raw_pickled_data_path}...")
        try:
            with open(raw_pickled_data_path, 'rb') as f:
                datasets = pickle.load(f)
            print("Raw datasets loaded from pickle successfully.")
        except Exception as e:
            print(f"Error loading raw datasets from pickle: {e}. Loading from CSVs instead...")
            datasets = load_all_data(data_folder)
    else:
        print("Raw datasets pickle not found. Loading from CSVs...")
        datasets = load_all_data(data_folder)

    # Assign to descriptive names from the loaded dictionary
    df_transaction_data = datasets.get('transaction_data')
    df_user_data = datasets.get('user_data')
    df_city_data = datasets.get('city_data')
    df_type_data = datasets.get('type_data')
    df_visit_mode_data = datasets.get('visit_mode_data')
    df_continent_data = datasets.get('continent_data')
    df_country_data = datasets.get('country_data')
    df_region_data = datasets.get('region_data')
    df_item_data = datasets.get('item_data') # This is Updated_Item.csv

    # Basic check if all essential datasets are loaded
    if not all([df_transaction_data is not None, df_user_data is not None, df_item_data is not None,
                df_city_data is not None, df_type_data is not None, df_visit_mode_data is not None,
                df_continent_data is not None, df_country_data is not None, df_region_data is not None]):
        print("Error: One or more essential datasets could not be loaded. Exiting preprocessing.")
        return None

    # --- Data Cleaning and Type Conversion ---
    print("Starting data cleaning and type conversion...")

    # Clean Region.csv and Continent.csv (from placeholder rows)
    df_region_data = df_region_data[~((df_region_data['Region'].isna()) | (df_region_data['Region'] == '-')) | (df_region_data['RegionId'] == 0.0)].copy()
    df_continent_data = df_continent_data[~((df_continent_data['Continent'].isna()) | (df_continent_data['Continent'] == '-')) | (df_continent_data['ContinentId'] == 0.0)].copy()

    # Handle missing 'AttractionTypeId' in df_item_data
    max_type_id = df_type_data['AttractionTypeId'].max()
    unknown_type_id = 999
    df_item_data['AttractionTypeId'] = pd.to_numeric(df_item_data['AttractionTypeId'], errors='coerce').astype('Int64') # Use nullable Int64 first
    df_item_data['AttractionTypeId'] = df_item_data['AttractionTypeId'].fillna(unknown_type_id).astype(int)
    if unknown_type_id not in df_type_data['AttractionTypeId'].values:
        df_type_data = pd.concat([df_type_data, pd.DataFrame([{'AttractionTypeId': unknown_type_id, 'AttractionType': 'Unknown'}])], ignore_index=True)

    # CRITICAL FIX: Handle missing 'CityId' in df_user_data (fill with 0)
    df_user_data['CityId'] = df_user_data['CityId'].fillna(0).astype(int)

    # Convert other relevant ID columns to integer type (if they aren't already)
    df_item_data['AttractionCityId'] = pd.to_numeric(df_item_data['AttractionCityId'], errors='coerce').astype(int)
    df_transaction_data['VisitMode'] = pd.to_numeric(df_transaction_data['VisitMode'], errors='coerce').astype(int)
    df_transaction_data['AttractionId'] = pd.to_numeric(df_transaction_data['AttractionId'], errors='coerce').astype(int)
    df_user_data['ContinentId'] = pd.to_numeric(df_user_data['ContinentId'], errors='coerce').astype(int)
    df_user_data['RegionId'] = pd.to_numeric(df_user_data['RegionId'], errors='coerce').astype(int)
    df_user_data['CountryId'] = pd.to_numeric(df_user_data['CountryId'], errors='coerce').astype(int)

    # --- Merging DataFrames ---
    print("Merging datasets...")

    # 1. Transaction + User
    df_main = pd.merge(df_transaction_data, df_user_data, on='UserId', how='left')
    df_main.rename(columns={
        'ContinentId': 'User_ContinentId',
        'RegionId': 'User_RegionId',
        'CountryId': 'User_CountryId',
        'CityId': 'User_CityId'
    }, inplace=True)

    # 2. Main + Item
    df_main = pd.merge(df_main, df_item_data, on='AttractionId', how='left')

    # 3. Main + City (for attraction's city)
    df_main = pd.merge(df_main, df_city_data, left_on='AttractionCityId', right_on='CityId', how='left')
    df_main.rename(columns={'CityName': 'Attraction_CityName', 'CountryId': 'Attraction_CountryId_FK'}, inplace=True)
    df_main.drop(columns=['CityId_y'], errors='ignore', inplace=True)
    df_main.rename(columns={'CityId_x': 'AttractionCityId_FK'}, inplace=True)

    # 4. Main + Type (for attraction's type)
    df_main = pd.merge(df_main, df_type_data, on='AttractionTypeId', how='left')

    # 5. Main + Visit Mode
    df_main = pd.merge(df_main, df_visit_mode_data, left_on='VisitMode', right_on='VisitModeId', how='left')
    df_main.rename(columns={'VisitMode_x': 'VisitMode_FK', 'VisitMode_y': 'VisitMode_Name'}, inplace=True)
    df_main.drop(columns=['VisitModeId'], errors='ignore', inplace=True)

    # 6. Main + Country (for user's country)
    df_main = pd.merge(df_main, df_country_data, left_on='User_CountryId', right_on='CountryId', how='left')
    df_main.rename(columns={'Country': 'User_Country_Name', 'RegionId': 'User_RegionId_FK_from_Country'}, inplace=True)
    df_main.drop(columns=['CountryId_y'], errors='ignore', inplace=True)
    df_main.rename(columns={'CountryId_x': 'User_CountryId_FK'}, inplace=True)

    # 7. Main + Region (for user's region)
    df_main = pd.merge(df_main, df_region_data, left_on='User_RegionId', right_on='RegionId', how='left')
    df_main.rename(columns={'Region': 'User_Region_Name', 'ContinentId': 'User_ContinentId_FK_from_Region'}, inplace=True)
    df_main.drop(columns=['RegionId_y'], errors='ignore', inplace=True)
    df_main.rename(columns={'RegionId_x': 'User_RegionId_FK'}, inplace=True)

    # 8. Main + Continent (for user's continent)
    df_main = pd.merge(df_main, df_continent_data, left_on='User_ContinentId', right_on='ContinentId', how='left')
    df_main.rename(columns={'Continent': 'User_Continent_Name'}, inplace=True)
    df_main.drop(columns=['ContinentId_y'], errors='ignore', inplace=True)
    df_main.rename(columns={'ContinentId_x': 'User_ContinentId_FK'}, inplace=True)

    # Drop any truly redundant ID columns after all merges that are still present
    df_main.drop(columns=[
        'AttractionCityId', # This was the FK from Item to City
        'User_RegionId_FK_from_Country', # Redundant after proper region merge
        'User_ContinentId_FK_from_Region' # Redundant after proper continent merge
    ], errors='ignore', inplace=True)

    print("Merging complete. Checking for missing values in final DataFrame...")
    missing_values = df_main.isnull().sum()[df_main.isnull().sum() > 0]
    if not missing_values.empty:
        print("Warning: Missing values found in the final consolidated DataFrame:")
        print(missing_values)
    else:
        print("No missing values found in the final consolidated DataFrame.")
    
    # --- NEW: Create output folder and save in both formats ---
    if not os.path.exists(processed_output_folder):
        os.makedirs(processed_output_folder)
        print(f"Created folder: {processed_output_folder}")

    # Save as pickle
    try:
        with open(processed_pkl_path, 'wb') as f:
            pickle.dump(df_main, f)
        print(f"Processed data saved to {processed_pkl_path}")
    except Exception as e:
        print(f"Error saving processed data to {processed_pkl_path}: {e}")

    # Save as CSV
    try:
        df_main.to_csv(processed_csv_path, index=False, encoding='utf-8')
        print(f"Processed data saved to {processed_csv_path}")
    except Exception as e:
        print(f"Error saving processed data to {processed_csv_path}: {e}")

    return df_main

if __name__ == "__main__":
    print("--- Testing data_preprocessing.py ---")
    consolidated_df = preprocess_data()
    if consolidated_df is not None:
        print("\nConsolidated DataFrame Head:")
        print(consolidated_df.head())
        print("\nConsolidated DataFrame Info:")
        print(consolidated_df.info())
    print("--- End of data_preprocessing.py test ---")
