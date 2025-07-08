# data_loading.py
import pandas as pd
import os
import pickle

def load_all_data(data_folder='Data'):
    """
    Loads all necessary CSV data files into pandas DataFrames.

    Args:
        data_folder (str): The name of the folder containing the CSV files.

    Returns:
        dict: A dictionary where keys are descriptive names of the datasets
              and values are their respective pandas DataFrames.
    """
    data_files = {
        'transaction_data': 'Transaction.csv',
        'user_data': 'User.csv',
        'city_data': 'City.csv',
        'type_data': 'Type.csv',
        'visit_mode_data': 'Mode.csv',
        'continent_data': 'Continent.csv',
        'country_data': 'Country.csv',
        'region_data': 'Region.csv',
        'item_data': 'Updated_Item.csv' # Using the updated item data
    }

    loaded_data = {}
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Expected data folder: {os.path.abspath(data_folder)}")

    for name, filename in data_files.items():
        file_path = os.path.join(data_folder, filename)
        print(f"Attempting to load: {os.path.abspath(file_path)}")

        try:
            # --- MODIFICATION HERE: Add encoding parameter ---
            loaded_data[name] = pd.read_csv(file_path, encoding='latin1') # Try latin1 first
            # If latin1 doesn't work, you might try 'cp1252' or 'ISO-8859-1'
            # loaded_data[name] = pd.read_csv(file_path, encoding='cp1252')
            # --- END MODIFICATION ---
            print(f"Successfully loaded {filename}")
        except FileNotFoundError:
            print(f"Error: {filename} not found at {os.path.abspath(file_path)}. Please ensure it's in the '{data_folder}' directory.")
            loaded_data[name] = None
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            loaded_data[name] = None

    return loaded_data

def save_datasets_to_pickle(datasets_dict, output_folder='Pickled_Data', filename='raw_datasets.pkl'):
    """
    Saves a dictionary of pandas DataFrames to a pickle file.

    Args:
        datasets_dict (dict): A dictionary of pandas DataFrames to save.
        output_folder (str): The folder where the pickle file will be saved.
        filename (str): The name of the pickle file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    file_path = os.path.join(output_folder, filename)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(datasets_dict, f)
        print(f"Successfully saved all raw datasets to {file_path}")
    except Exception as e:
        print(f"Error saving raw datasets to {file_path}: {e}")

if __name__ == "__main__":
    print("--- Running data_loading.py ---")
    datasets = load_all_data()
    essential_datasets_loaded = all(df is not None for df in datasets.values())

    if essential_datasets_loaded:
        import sys
        if '--save_raw_pickle' in sys.argv:
            save_datasets_to_pickle(datasets)
        else:
            print("\nTo save raw datasets to a pickle file, run with '--save_raw_pickle' argument.")
            print("Example: python data_loading.py --save_raw_pickle")
            print("\nDisplaying head of Transaction Data (for quick check):")
            if datasets.get('transaction_data') is not None:
                print(datasets['transaction_data'].head())
    else:
        print("Not all essential datasets were loaded. Skipping save operation.")

    print("--- End of data_loading.py execution ---")
