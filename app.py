# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# For Recommendation System
from surprise import Dataset, Reader, SVD
from surprise.prediction_algorithms.predictions import Prediction

# --- Configuration ---
BASE_DIR = os.getcwd()
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
EDA_PLOTS_DIR = os.path.join(BASE_DIR, 'eda_plot')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'Data') # Need access to raw City.csv for full city names

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Function for Loading Assets ---
@st.cache_resource
def load_assets():
    """Loads all necessary models, preprocessors, and data."""
    assets = {}
    st.spinner("Loading models and data...")
    try:
        # Load Processed Data (main dataframe)
        processed_data_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.pkl')
        with open(processed_data_path, 'rb') as f:
            assets['df_main'] = pickle.load(f)
        
        # --- FIX: Clip Ratings to 1-5 Range ---
        assets['df_main']['Rating'] = assets['df_main']['Rating'].clip(1, 5)

        # Prepare df_main for consistent feature extraction (same as in model_training.py)
        assets['df_main'].dropna(subset=['Rating', 'VisitMode_Name'], inplace=True)

        # Convert ID columns to appropriate integer types (same as in model_training.py)
        id_cols_to_convert_int_actual = [
            'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId',
            'AttractionTypeId', 'CityId', 'VisitMode_FK'
        ]
        for col in id_cols_to_convert_int_actual:
            if col in assets['df_main'].columns:
                assets['df_main'][col] = pd.to_numeric(assets['df_main'][col], errors='coerce').astype('Int64')
                assets['df_main'][col] = assets['df_main'][col].fillna(-1).astype(int)

        # Load Regression Model and Scaler
        with open(os.path.join(MODELS_DIR, 'lightgbm_regressor.pkl'), 'rb') as f:
            assets['reg_model'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'regression_scaler.pkl'), 'rb') as f:
            assets['scaler_reg'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'regression_features.pkl'), 'rb') as f:
            assets['reg_features'] = pickle.load(f)

        # Load Classification Model and Label Encoder
        with open(os.path.join(MODELS_DIR, 'random_forest_classifier.pkl'), 'rb') as f:
            assets['clf_model'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'classification_label_encoder.pkl'), 'rb') as f:
            assets['label_encoder'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'classification_features.pkl'), 'rb') as f:
            assets['clf_features'] = pickle.load(f)

        # Load Recommendation Model
        with open(os.path.join(MODELS_DIR, 'svd_recommendation_model.pkl'), 'rb') as f:
            assets['algo_svd'] = pickle.load(f)
        
        # Prepare lookup DataFrames for UI dropdowns and display
        # FIX: Include 'AttractionTypeId' in attractions_lookup
        assets['attractions_lookup'] = assets['df_main'][['AttractionId', 'Attraction', 'AttractionType', 'Attraction_CityName', 'AttractionTypeId']].drop_duplicates().set_index('AttractionId')
        assets['users_lookup'] = assets['df_main'][['UserId', 'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId']].drop_duplicates().set_index('UserId')
        
        # Get unique values for dropdowns from df_main (ensure consistency with data types)
        assets['visit_modes'] = assets['df_main']['VisitMode_Name'].unique().tolist()
        assets['attraction_types'] = assets['df_main']['AttractionType'].unique().tolist()
        
        # --- FIX: Load raw data for comprehensive name lookups for cascading dropdowns ---
        df_city_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, 'City.csv'), encoding='latin1')
        df_country_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Country.csv'), encoding='latin1')
        df_region_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Region.csv'), encoding='latin1')
        df_continent_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Continent.csv'), encoding='latin1')

        # Clean raw Region and Continent data (same as in data_preprocessing.py)
        df_region_raw = df_region_raw[~((df_region_raw['Region'].isna()) | (df_region_raw['Region'] == '-')) | (df_region_raw['RegionId'] == 0.0)].copy()
        df_continent_raw = df_continent_raw[~((df_continent_raw['Continent'].isna()) | (df_continent_raw['Continent'] == '-')) | (df_continent_raw['ContinentId'] == 0.0)].copy()

        # Create comprehensive lookup for user's locations with names
        user_loc_lookup = assets['df_main'][['User_CityId', 'User_CountryId', 'User_RegionId', 'User_ContinentId']].drop_duplicates()

        # Merge with City names
        user_loc_lookup = user_loc_lookup.merge(
            df_city_raw[['CityId', 'CityName']],
            left_on='User_CityId',
            right_on='CityId',
            how='left'
        ).rename(columns={'CityName': 'User_City_Name'}).drop(columns=['CityId'])

        # Merge with Country names
        user_loc_lookup = user_loc_lookup.merge(
            df_country_raw[['CountryId', 'Country']],
            left_on='User_CountryId',
            right_on='CountryId',
            how='left'
        ).rename(columns={'Country': 'User_Country_Name'}).drop(columns=['CountryId'])

        # Merge with Region names
        user_loc_lookup = user_loc_lookup.merge(
            df_region_raw[['RegionId', 'Region']],
            left_on='User_RegionId',
            right_on='RegionId',
            how='left'
        ).rename(columns={'Region': 'User_Region_Name'}).drop(columns=['RegionId'])

        # Merge with Continent names
        user_loc_lookup = user_loc_lookup.merge(
            df_continent_raw[['ContinentId', 'Continent']],
            left_on='User_ContinentId',
            right_on='ContinentId',
            how='left'
        ).rename(columns={'Continent': 'User_Continent_Name'}).drop(columns=['ContinentId'])

        assets['user_cities_with_names'] = user_loc_lookup
        
        # Create mappings for IDs to Names for user input and display
        assets['continent_name_to_id'] = assets['df_main'][['User_Continent_Name', 'User_ContinentId']].drop_duplicates().set_index('User_Continent_Name')['User_ContinentId'].to_dict()
        assets['country_name_to_id'] = assets['df_main'][['User_Country_Name', 'User_CountryId']].drop_duplicates().set_index('User_Country_Name')['User_CountryId'].to_dict()
        assets['region_name_to_id'] = assets['df_main'][['User_Region_Name', 'User_RegionId']].drop_duplicates().set_index('User_Region_Name')['User_RegionId'].to_dict()
        
        assets['attraction_type_name_to_id'] = assets['df_main'][['AttractionType', 'AttractionTypeId']].drop_duplicates().set_index('AttractionType')['AttractionTypeId'].to_dict()
        assets['visit_mode_name_to_id'] = assets['df_main'][['VisitMode_Name', 'VisitMode_FK']].drop_duplicates().set_index('VisitMode_Name')['VisitMode_FK'].to_dict()
        assets['attraction_city_name_to_id'] = assets['df_main'][['Attraction_CityName', 'CityId']].drop_duplicates().set_index('Attraction_CityName')['CityId'].to_dict()

        # Add City ID to Name lookup for display of User City
        assets['city_id_to_name'] = df_city_raw.set_index('CityId')['CityName'].to_dict()
        # FIX: Ensure this reverse mapping for city name to ID is correctly created
        assets['city_name_to_id'] = {v: k for k, v in assets['city_id_to_name'].items()} # Reverse mapping

        # FIX: Ensure these top-level lists are correctly populated for selectboxes
        assets['user_continents_all'] = assets['df_main']['User_Continent_Name'].unique().tolist()
        assets['user_countries_all'] = assets['df_main']['User_Country_Name'].unique().tolist()
        assets['user_regions_all'] = assets['df_main']['User_Region_Name'].unique().tolist()
        assets['attraction_cities_all'] = assets['df_main']['Attraction_CityName'].unique().tolist()


        st.success("All assets loaded successfully!")
        return assets
    
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Please ensure all previous scripts (data_preprocessing.py, model_training.py, eda.py) were run successfully. Missing: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading assets: {e}")
        st.exception(e)
        st.stop()

# Load all assets once at the start of the app
assets = load_assets()

# Assign loaded assets to variables for easier access
df_main = assets['df_main']
reg_model = assets['reg_model']
scaler_reg = assets['scaler_reg']
reg_features = assets['reg_features']
clf_model = assets['clf_model']
label_encoder = assets['label_encoder']
clf_features = assets['clf_features']
algo_svd = assets['algo_svd']
attractions_lookup = assets['attractions_lookup']
users_lookup = assets['users_lookup']
visit_modes = assets['visit_modes']
attraction_types = assets['attraction_types']
user_continents_all = assets['user_continents_all']
user_countries_all = assets['user_countries_all']
user_regions_all = assets['user_regions_all']
attraction_cities_all = assets['attraction_cities_all']

# Mappings and lookups
continent_name_to_id = assets['continent_name_to_id']
country_name_to_id = assets['country_name_to_id']
region_name_to_id = assets['region_name_to_id']
attraction_type_name_to_id = assets['attraction_type_name_to_id']
visit_mode_name_to_id = assets['visit_mode_name_to_id']
attraction_city_name_to_id = assets['attraction_city_name_to_id']
user_cities_with_names = assets['user_cities_with_names']
city_id_to_name = assets['city_id_to_name']
city_name_to_id = assets['city_name_to_id']


# --- Helper Function for Feature Preparation for Models ---
def prepare_input_features(user_input, features_list, is_regression_model=True):
    """
    Prepares a DataFrame from user inputs, aligns columns with model's expected features,
    and applies scaling/categorical dtype conversion.
    """
    input_df = pd.DataFrame([user_input])

    for col in features_list:
        if col not in input_df.columns:
            input_df[col] = -1

    input_df = input_df[features_list]

    categorical_cols_for_dt = [
        'VisitMonth', 'User_ContinentId', 'User_RegionId', 'User_CountryId', 'User_CityId',
        'AttractionTypeId', 'CityId', 'VisitMode_FK'
    ]
    
    categorical_features_for_dt_existing = [
        col for col in categorical_cols_for_dt if col in input_df.columns
    ]

    for col in categorical_features_for_dt_existing:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(-1).astype(int)
        if is_regression_model:
            input_df[col] = input_df[col].astype('category')


    numerical_features_to_scale = ['VisitYear']
    numerical_features_to_scale_existing = [col for col in numerical_features_to_scale if col in input_df.columns]

    if numerical_features_to_scale_existing:
        input_df[numerical_features_to_scale_existing] = scaler_reg.transform(input_df[numerical_features_to_scale_existing])
    
    return input_df


# --- Streamlit UI ---
st.title("🗺️ Tourism Experience Analytics")
st.markdown("### Explore, Predict & Recommend")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Predict Visit Mode", "Predict Attraction Rating", "Get Recommendations", "Explore Data (EDA)"])

if page == "Project Overview":
    st.header("Project Overview")
    st.write("""
        This application provides insights and predictions for the tourism industry,
        enhancing user experiences by leveraging machine learning and data analytics.
        It offers personalized recommendations, predicts user satisfaction, and classifies
        potential user behavior based on historical travel patterns and attraction features.
    """)
    st.subheader("Key Features:")
    st.markdown("""
    -   **Predict User Visit Mode:** Forecast the purpose of a user's visit (e.g., Business, Family).
    -   **Predict Attraction Rating:** Estimate the rating a user is likely to give an attraction.
    -   **Personalized Recommendations:** Suggest attractions tailored to individual user preferences.
    -   **Interactive Analytics Dashboard:** Visualize popular attractions, user demographics, and travel patterns.
    """)

elif page == "Predict Visit Mode":
    st.header("👤 Predict User Visit Mode")
    st.write("Predict the likely mode of a user's visit (e.g., Business, Family, Couples, Friends) based on their details.")

    with st.form("visit_mode_form"):
        st.subheader("User & Visit Details")
        col1, col2 = st.columns(2)
        
        with col1:
            # Cascading dropdowns for User's Location
            selected_continent_name_clf = st.selectbox("User Continent", user_continents_all, key='user_cont_clf')
            
            # Filter countries based on selected continent
            filtered_countries_df_clf = user_cities_with_names[
                (user_cities_with_names['User_ContinentId'] == continent_name_to_id.get(selected_continent_name_clf, -1))
            ].drop_duplicates(subset=['User_Country_Name'])
            filtered_countries_clf = filtered_countries_df_clf['User_Country_Name'].unique().tolist()
            
            selected_country_name_clf = st.selectbox("User Country", filtered_countries_clf, key='user_country_clf')
            
            # Filter regions based on selected country
            filtered_regions_df_clf = user_cities_with_names[
                (user_cities_with_names['User_CountryId'] == country_name_to_id.get(selected_country_name_clf, -1))
            ].drop_duplicates(subset=['User_Region_Name'])
            filtered_regions_clf = filtered_regions_df_clf['User_Region_Name'].unique().tolist()
            
            selected_region_name_clf = st.selectbox("User Region", filtered_regions_clf, key='user_region_clf')
            
            # Filter cities based on selected region
            filtered_cities_df_clf = user_cities_with_names[
                (user_cities_with_names['User_RegionId'] == region_name_to_id.get(selected_region_name_clf, -1))
            ].drop_duplicates(subset=['User_City_Name'])
            filtered_cities_clf = filtered_cities_df_clf['User_City_Name'].unique().tolist()
            
            selected_user_city_name_clf = st.selectbox("User City", filtered_cities_clf, key='user_city_clf')

        with col2:
            visit_year = st.number_input("Visit Year", min_value=2000, max_value=2030, value=2022, step=1, key='visit_year_clf')
            visit_month = st.number_input("Visit Month (1-12)", min_value=1, max_value=12, value=10, step=1, key='visit_month_clf')
            attraction_type_name = st.selectbox("Attraction Type (Visited)", attraction_types, key='attraction_type_clf')
            attraction_city_name = st.selectbox("Attraction City (Visited)", attraction_cities_all, key='attraction_city_clf')

        submitted_clf = st.form_submit_button("Predict Visit Mode")

    if submitted_clf:
        user_input_clf = {
            'VisitYear': visit_year,
            'VisitMonth': visit_month,
            'User_ContinentId': continent_name_to_id.get(selected_continent_name_clf, -1),
            'User_RegionId': region_name_to_id.get(selected_region_name_clf, -1),
            'User_CountryId': country_name_to_id.get(selected_country_name_clf, -1),
            'User_CityId': city_name_to_id.get(selected_user_city_name_clf, -1),
            'AttractionTypeId': attraction_type_name_to_id.get(attraction_type_name, -1),
            'CityId': attraction_city_name_to_id.get(attraction_city_name, -1),
        }
        
        processed_input_clf = prepare_input_features(user_input_clf, clf_features, is_regression_model=False)

        try:
            prediction_encoded = clf_model.predict(processed_input_clf)
            predicted_visit_mode = label_encoder.inverse_transform(prediction_encoded)
            st.success(f"**Predicted Visit Mode:** {predicted_visit_mode[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Please check input values and ensure all necessary features are available.")
            st.write(f"Input features created: {processed_input_clf.columns.tolist()}")


elif page == "Predict Attraction Rating":
    st.header("⭐ Predict Attraction Rating")
    st.write("Estimate the rating a user might give to a specific tourist attraction.")

    with st.form("rating_form"):
        st.subheader("User, Visit & Attraction Details")
        col1, col2 = st.columns(2)

        with col1:
            selected_continent_name_reg = st.selectbox("User Continent", user_continents_all, key='user_cont_reg')
            
            filtered_countries_df_reg = user_cities_with_names[
                (user_cities_with_names['User_ContinentId'] == continent_name_to_id.get(selected_continent_name_reg, -1))
            ].drop_duplicates(subset=['User_Country_Name'])
            filtered_countries_reg = filtered_countries_df_reg['User_Country_Name'].unique().tolist()
            
            selected_country_name_reg = st.selectbox("User Country", filtered_countries_reg, key='user_country_reg')
            
            filtered_regions_df_reg = user_cities_with_names[
                (user_cities_with_names['User_CountryId'] == country_name_to_id.get(selected_country_name_reg, -1))
            ].drop_duplicates(subset=['User_Region_Name'])
            filtered_regions_reg = filtered_regions_df_reg['User_Region_Name'].unique().tolist()
            
            selected_region_name_reg = st.selectbox("User Region", filtered_regions_reg, key='user_region_reg')
            
            filtered_cities_df_reg = user_cities_with_names[
                (user_cities_with_names['User_RegionId'] == region_name_to_id.get(selected_region_name_reg, -1))
            ].drop_duplicates(subset=['User_City_Name'])
            filtered_cities_reg = filtered_cities_df_reg['User_City_Name'].unique().tolist()
            
            selected_user_city_name_reg = st.selectbox("User City", filtered_cities_reg, key='user_city_reg')

            visit_mode_name = st.selectbox("Visit Mode", visit_modes, key='visit_mode_reg')
        
        with col2:
            visit_year = st.number_input("Visit Year", min_value=2000, max_value=2030, value=2022, step=1, key='visit_year_reg')
            visit_month = st.number_input("Visit Month (1-12)", min_value=1, max_value=12, value=10, step=1, key='visit_month_reg')
            attraction_name = st.selectbox("Attraction to Rate", attractions_lookup['Attraction'].unique().tolist(), key='attraction_reg')
            
            selected_attraction_info = attractions_lookup[attractions_lookup['Attraction'] == attraction_name]
            attraction_type_id_from_selection = selected_attraction_info['AttractionTypeId'].iloc[0] if not selected_attraction_info.empty else -1
            attraction_city_name_from_selection = selected_attraction_info['Attraction_CityName'].iloc[0] if not selected_attraction_info.empty else "Unknown"
            

        submitted_reg = st.form_submit_button("Predict Rating")

    if submitted_reg:
        user_input_reg = {
            'VisitYear': visit_year,
            'VisitMonth': visit_month,
            'User_ContinentId': continent_name_to_id.get(selected_continent_name_reg, -1),
            'User_RegionId': region_name_to_id.get(selected_region_name_reg, -1),
            'User_CountryId': country_name_to_id.get(selected_country_name_reg, -1),
            'User_CityId': city_name_to_id.get(selected_user_city_name_reg, -1),
            'AttractionTypeId': attraction_type_id_from_selection,
            'CityId': attraction_city_name_to_id.get(attraction_city_name_from_selection, -1),
            'VisitMode_FK': visit_mode_name_to_id.get(visit_mode_name, -1)
        }
        
        processed_input_reg = prepare_input_features(user_input_reg, reg_features, is_regression_model=True)

        try:
            predicted_rating = reg_model.predict(processed_input_reg)[0]
            st.success(f"**Predicted Rating:** {int(round(predicted_rating))} out of 5")
            if predicted_rating < 3.0:
                st.info("💡 This attraction might receive a lower rating. Consider improvements or adjust expectations.")
            elif predicted_rating >= 4.5:
                st.info("🌟 This attraction is likely to be highly rated!")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Please check input values and ensure all necessary features are available.")
            st.write(f"Input features created: {processed_input_reg.columns.tolist()}")


elif page == "Get Recommendations":
    st.header("✨ Personalized Attraction Suggestions")
    st.write("Receive a ranked list of recommended attractions based on a user's historical preferences and similar users’ preferences.")

    available_user_ids = df_main['UserId'].unique().tolist()
    
    user_id_input = st.selectbox("Select a User ID for Recommendations", available_user_ids, key='rec_user_id')
    num_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10, step=1)

    if st.button("Generate Recommendations"):
        if user_id_input not in df_main['UserId'].unique():
            st.warning("Please select a valid User ID from the dataset.")
        else:
            st.subheader(f"Recommendations for User ID: {user_id_input}")

            with st.spinner("Generating recommendations..."):
                visited_attraction_ids = df_main[df_main['UserId'] == user_id_input]['AttractionId'].unique()

                all_attraction_ids = df_main['AttractionId'].unique()

                unvisited_attraction_ids = [
                    att_id for att_id in all_attraction_ids
                    if att_id not in visited_attraction_ids
                ]

                predictions_for_user = []
                for attraction_id in unvisited_attraction_ids:
                    predicted_rating = algo_svd.predict(user_id_input, attraction_id).est
                    predictions_for_user.append({'AttractionId': attraction_id, 'PredictedRating': predicted_rating})

                predictions_df = pd.DataFrame(predictions_for_user)
                
                if predictions_df.empty:
                    st.info("No unvisited attractions to recommend for this user, or user has no historical data for recommendations.")
                else:
                    top_recommendations = predictions_df.sort_values(by='PredictedRating', ascending=False).head(num_recommendations)

                    recommended_attractions_details = top_recommendations.merge(
                        attractions_lookup.reset_index(),
                        on='AttractionId',
                        how='left'
                    )

                    if not recommended_attractions_details.empty:
                        recommended_attractions_details['PredictedRating'] = recommended_attractions_details['PredictedRating'].round(0).astype(int)
                        st.table(recommended_attractions_details[['Attraction', 'AttractionType', 'Attraction_CityName', 'PredictedRating']])
                    else:
                        st.info("No recommendations generated based on the criteria.")


elif page == "Explore Data (EDA)":
    st.header("📊 Explore Data (EDA)")
    st.write("View key insights and visualizations from the Exploratory Data Analysis phase.")

    eda_plots = [
        'user_distribution_continent.png',
        'user_distribution_top_countries.png',
        'user_distribution_top_regions.png',
        'attraction_type_popularity_count.png',
        'attraction_type_avg_rating.png',
        'top_10_attractions_by_avg_rating.png',
        'overall_rating_distribution.png',
        'rating_distribution_by_region.png',
        'visit_mode_by_continent_counts.png',
        'visit_mode_by_continent_proportions.png',
    ]

    for plot_filename in eda_plots:
        plot_path = os.path.join(EDA_PLOTS_DIR, plot_filename)
        if os.path.exists(plot_path):
            st.subheader(plot_filename.replace('_', ' ').replace('.png', '').title())
            st.image(plot_path, use_column_width=True)
        else:
            st.warning(f"Plot '{plot_filename}' not found at '{plot_path}'. Please ensure eda.py was run.")

st.sidebar.markdown("---")
st.sidebar.info("Developed for Tourism Experience Analytics Project")
