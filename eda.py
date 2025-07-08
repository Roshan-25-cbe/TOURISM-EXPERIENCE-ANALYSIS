# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from data_preprocessing import preprocess_data # To load the processed data

def perform_eda(plots_output_folder='eda_plot'):
    """
    Performs Exploratory Data Analysis (EDA) on the consolidated DataFrame
    and saves the generated plots.

    Args:
        plots_output_folder (str): The folder where EDA plots will be saved.
    """
    print("--- Starting Exploratory Data Analysis (EDA) ---")

    # Load the processed data using preprocess_data function.
    df_main = preprocess_data()

    if df_main is None:
        print("Failed to load processed data. Exiting EDA.")
        return

    # Create the plots output folder if it doesn't exist
    if not os.path.exists(plots_output_folder):
        os.makedirs(plots_output_folder)
        print(f"Created plots folder: {plots_output_folder}")

    # --- EDA Objective 1: Visualize user distribution across continents, countries, and regions ---
    print("\n--- Visualizing User Distribution ---")

    # User Distribution by Continent
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_main, y='User_Continent_Name', order=df_main['User_Continent_Name'].value_counts().index, palette='viridis')
    plt.title('User Distribution by Continent')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Continent')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'user_distribution_continent.png'))
    print("Plot 'user_distribution_continent.png' saved.")
    plt.close()

    # User Distribution by Country (Top N countries for better visualization)
    plt.figure(figsize=(12, 8))
    top_countries = df_main['User_Country_Name'].value_counts().nlargest(20).index
    sns.countplot(data=df_main[df_main['User_Country_Name'].isin(top_countries)], y='User_Country_Name', order=top_countries, palette='cividis')
    plt.title('Top 20 User Distribution by Country')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'user_distribution_top_countries.png'))
    print("Plot 'user_distribution_top_countries.png' saved.")
    plt.close()

    # User Distribution by Region (Top N regions for better visualization)
    plt.figure(figsize=(12, 8))
    top_regions = df_main['User_Region_Name'].value_counts().nlargest(20).index
    sns.countplot(data=df_main[df_main['User_Region_Name'].isin(top_regions)], y='User_Region_Name', order=top_regions, palette='magma')
    plt.title('Top 20 User Distribution by Region')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Region')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'user_distribution_top_regions.png'))
    print("Plot 'user_distribution_top_regions.png' saved.")
    plt.close()

    # --- EDA Objective 2: Explore attraction types and their popularity based on user ratings ---
    print("\n--- Exploring Attraction Types and Popularity ---")

    # Attraction Type Popularity (Count)
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df_main, y='AttractionType', order=df_main['AttractionType'].value_counts().index, palette='cubehelix')
    plt.title('Attraction Type Popularity (Number of Visits)')
    plt.xlabel('Number of Visits')
    plt.ylabel('Attraction Type')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'attraction_type_popularity_count.png'))
    print("Plot 'attraction_type_popularity_count.png' saved.")
    plt.close()

    # Attraction Type Average Rating
    avg_rating_by_type = df_main.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=avg_rating_by_type.values, y=avg_rating_by_type.index, palette='plasma')
    plt.title('Average Rating by Attraction Type')
    plt.xlabel('Average Rating (1-5)')
    plt.ylabel('Attraction Type')
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'attraction_type_avg_rating.png'))
    print("Plot 'attraction_type_avg_rating.png' saved.")
    plt.close()

    # Top 10 Attractions by Average Rating (Ensure at least 5 ratings for meaningful average)
    min_ratings_for_avg = 5
    attraction_ratings = df_main.groupby('Attraction').agg(
        avg_rating=('Rating', 'mean'),
        num_ratings=('Rating', 'count')
    ).reset_index()
    top_10_avg_rated_attractions = attraction_ratings[attraction_ratings['num_ratings'] >= min_ratings_for_avg].sort_values(by='avg_rating', ascending=False).head(10)

    if not top_10_avg_rated_attractions.empty:
        plt.figure(figsize=(12, 7))
        sns.barplot(x='avg_rating', y='Attraction', data=top_10_avg_rated_attractions, palette='rocket')
        plt.title(f'Top 10 Attractions by Average Rating (min {min_ratings_for_avg} ratings)')
        plt.xlabel('Average Rating (1-5)')
        plt.ylabel('Attraction Name')
        plt.xlim(0, 5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_output_folder, 'top_10_attractions_by_avg_rating.png'))
        print("Plot 'top_10_attractions_by_avg_rating.png' saved.")
        plt.close()
    else:
        print("Not enough attractions with sufficient ratings to plot Top 10 Average Rated Attractions.")


    # --- EDA Objective 3: Analyze distribution of ratings across different attractions and regions ---
    print("\n--- Analyzing Rating Distribution ---")

    # Overall Rating Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_main, x='Rating', palette='crest')
    plt.title('Overall Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Number of Ratings')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'overall_rating_distribution.png'))
    print("Plot 'overall_rating_distribution.png' saved.")
    plt.close()

    # Rating Distribution by Region (Top N regions)
    plt.figure(figsize=(14, 8))
    top_regions_in_df = df_main['User_Region_Name'].value_counts().nlargest(20).index
    sns.countplot(data=df_main[df_main['User_Region_Name'].isin(top_regions_in_df)], x='Rating', hue='User_Region_Name', palette='tab10')
    plt.title('Rating Distribution by Top Regions')
    plt.xlabel('Rating')
    plt.ylabel('Number of Ratings')
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'rating_distribution_by_region.png'))
    print("Plot 'rating_distribution_by_region.png' saved.")
    plt.close()


    # --- NEW EDA Objective: Investigate correlation between Visit Mode and user demographics ---
    print("\n--- Investigating Visit Mode and User Demographics ---")

    # Visit Mode Distribution by Continent (Stacked Bar Chart)
    visit_mode_continent_counts = df_main.groupby(['User_Continent_Name', 'VisitMode_Name']).size().unstack(fill_value=0)
    visit_mode_continent_counts_norm = visit_mode_continent_counts.apply(lambda x: x / x.sum(), axis=1)

    plt.figure(figsize=(12, 7))
    visit_mode_continent_counts.plot(kind='bar', stacked=True, colormap='Paired', ax=plt.gca())
    plt.title('Visit Mode Distribution by Continent (Raw Counts)')
    plt.xlabel('Continent')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Visit Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'visit_mode_by_continent_counts.png'))
    print("Plot 'visit_mode_by_continent_counts.png' saved.")
    plt.close()

    plt.figure(figsize=(12, 7))
    visit_mode_continent_counts_norm.plot(kind='bar', stacked=True, colormap='Paired', ax=plt.gca())
    plt.title('Visit Mode Distribution by Continent (Proportions)')
    plt.xlabel('Continent')
    plt.ylabel('Proportion of Transactions')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Visit Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'visit_mode_by_continent_proportions.png'))
    print("Plot 'visit_mode_by_continent_proportions.png' saved.")
    plt.close()

    # Visit Mode Distribution by Top Countries (Stacked Bar Chart)
    df_top_countries = df_main[df_main['User_Country_Name'].isin(top_countries)]
    visit_mode_country_counts = df_top_countries.groupby(['User_Country_Name', 'VisitMode_Name']).size().unstack(fill_value=0)
    visit_mode_country_counts_norm = visit_mode_country_counts.apply(lambda x: x / x.sum(), axis=1)

    plt.figure(figsize=(14, 8))
    visit_mode_country_counts.plot(kind='bar', stacked=True, colormap='Paired', ax=plt.gca())
    plt.title('Visit Mode Distribution by Top Countries (Raw Counts)')
    plt.xlabel('Country')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=90, ha='right')
    plt.legend(title='Visit Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'visit_mode_by_country_counts.png'))
    print("Plot 'visit_mode_by_country_counts.png' saved.")
    plt.close()

    plt.figure(figsize=(14, 8))
    visit_mode_country_counts_norm.plot(kind='bar', stacked=True, colormap='Paired', ax=plt.gca())
    plt.title('Visit Mode Distribution by Top Countries (Proportions)')
    plt.xlabel('Country')
    plt.ylabel('Proportion of Transactions')
    plt.xticks(rotation=90, ha='right')
    plt.legend(title='Visit Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_folder, 'visit_mode_by_country_proportions.png'))
    print("Plot 'visit_mode_by_country_proportions.png' saved.")
    plt.close()

    print("\n--- All EDA Objectives Completed ---")
    print(f"All generated plots are saved in the '{plots_output_folder}' folder.")


if __name__ == "__main__":
    perform_eda()
    print("--- End of eda.py execution ---")
