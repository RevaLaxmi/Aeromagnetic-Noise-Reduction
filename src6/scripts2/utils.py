# utils.py

def extract_features_and_labels(df, target='mag_1_c'):
    """
    Extracts features and labels from the DataFrame for model training.
    
    Args:
    df (pd.DataFrame): The cleaned DataFrame after preprocessing.
    target (str): The column name to be used as the target variable.
    
    Returns:
    X (pd.DataFrame): The feature set.
    y (pd.Series): The target variable.
    """
    
    # Select the relevant features based on your provided list
    features = [
        'mag_1_c', 'mag_1_dc', 'mag_1_lag', 'mag_1_uc', 
        'lat', 'lon', 'utm_x', 'utm_y',
        'ins_lat', 'ins_lon', 'ins_acc_x',
        'dem', 'diurnal', 'flux_a_t', 'flux_b_t', 'flux_a_x'
    ]
    
    # Extract features (X) and target variable (y)
    X = df[features].copy()
    y = df[target].copy()
    
    return X, y
