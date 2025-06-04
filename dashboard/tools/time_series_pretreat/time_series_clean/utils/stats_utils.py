import pandas as pd

# Extracted from time_series_clean_utils.py
def calculate_descriptive_stats(df_processed, selected_column_names_list):
    """
    Calculates descriptive statistics for selected columns of a DataFrame.

    Args:
        df_processed (pd.DataFrame): The DataFrame to analyze.
        selected_column_names_list (list): A list of column names (or objects) 
                                           for which to calculate statistics.

    Returns:
        pd.DataFrame | None: A DataFrame containing the descriptive statistics, 
                           or None if no valid columns were selected or found.
    """
    if not selected_column_names_list:
        return None 

    valid_columns_to_describe = [col for col in selected_column_names_list if col in df_processed.columns]

    if not valid_columns_to_describe:
        print("Warning: No valid columns found in the DataFrame for descriptive stats calculation.")
        return None 

    try:
        desc_stats_df = df_processed[valid_columns_to_describe].describe()
        return desc_stats_df
    except Exception as desc_e:
        print(f"Error calculating descriptive statistics: {desc_e}")
        return None 