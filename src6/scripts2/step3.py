'''
SELECTING THE FEAUTRES IN THIS STEP


understanding which ones work the ebst i nterms of what we should work with
their priority
'''


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
import h5py
from sklearn.ensemble import HistGradientBoostingRegressor



def load_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        # Initialize an empty dictionary to store the data
        data = {}

        # Iterate through the keys of the file
        for key in f.keys():
            dataset = f[key]
            # Check if the dataset is an array or scalar
            if dataset.ndim > 0:
                data[key] = dataset[:]  # Extract array data
            else:
                data[key] = dataset[()]  # Extract scalar data

    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame(data)
    return df


# Function to evaluate the model with a given set of features
def evaluate_model(df, features, target_column):
    # Prepare feature matrix X and target vector y
    X = df[features]
    y = df[target_column]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling (important for models like RandomForest)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    model = HistGradientBoostingRegressor(random_state=42) # takes out Nan values on its own 
    model.fit(X_train_scaled, y_train)

    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

# Recursive feature selection function
def recursive_feature_selection(df, available_features, target_column, max_features=11):
    best_features = []
    best_r2 = -np.inf  # Initialize with very low r2
    feature_sets_to_test = []

    # Start by testing one feature at a time
    for feature in available_features:
        feature_sets_to_test.append([feature])
    
    # Evaluate performance with each set of features
    for feature_set in feature_sets_to_test:
        mse, r2 = evaluate_model(df, feature_set, target_column)
        print(f"Evaluating with features: {feature_set} => MSE: {mse:.3f}, R²: {r2:.3f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_features = feature_set
    
    # Add features incrementally and evaluate
    for num_features in range(2, max_features + 1):
        # Generate combinations of features with num_features
        feature_combinations = list(combinations(available_features, num_features))
        
        for combo in feature_combinations:
            mse, r2 = evaluate_model(df, combo, target_column)
            print(f"Evaluating with features: {combo} => MSE: {mse:.3f}, R²: {r2:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_features = combo
    
    return best_features, best_r2

# Function to save the best features to a CSV file for later testing
def save_best_feature_set(df, best_features, file_name="best_feature_set.csv"):
    df_subset = df[list(best_features)]
    df_subset.to_csv(file_name, index=False)
    print(f"Best feature set saved to {file_name}")

# Main function to run the feature selection process
def main():
    # Path to your H5 file
    h5_file_path = "data/Flt1003_train.h5"
    
    # Load the data
    df = load_data(h5_file_path)
    
    # Target column (replace with your actual target column name)
    target_column = 'mag_1_c'  # Adjust as needed
    
    # List of all available features
    available_features = [
        'cur_ac_lo', 'cur_acpwr', 'cur_bat_1', 'cur_bat_2', 
        'cur_com_1', 'cur_flap', 'cur_heat', 'cur_outpwr', 'cur_srvo_i', 'cur_srvo_m', 
        'cur_srvo_o', 'cur_strb', 'cur_tank', 'dem', 'diurnal', 'drape', 'dt', 'flight', 
        'flux_a_t', 'flux_a_x', 'flux_a_y', 'flux_a_z', 'flux_b_t', 'flux_b_x', 'flux_b_y', 
        'flux_b_z', 'flux_c_t', 'flux_c_x', 'flux_c_y', 'flux_c_z', 'flux_d_t', 'flux_d_x', 
        'flux_d_y', 'flux_d_z', 'ins_acc_x', 'ins_acc_y', 'ins_acc_z', 'ins_alt', 'ins_lat', 
        'ins_lon', 'ins_pitch', 'ins_roll', 'ins_vn', 'ins_vu', 'ins_vw', 'ins_wander', 
        'ins_yaw', 'lat', 'lgtl_acc', 'line', 'lon', 'ltrl_acc', 'mag_1_c', 'mag_1_dc', 
        'mag_1_igrf', 'mag_1_lag', 'mag_1_uc', 'mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
        'msl', 'nrml_acc', 'ogs_alt', 'ogs_mag', 'pitch_rate', 'pitot_p', 'radar', 'roll_rate', 
        'static_p', 'tas', 'topo', 'total_p', 'tt', 'utm_x', 'utm_y', 'utm_z', 'vol_acc_n', 
        'vol_acc_p', 'vol_acpwr', 'vol_back', 'vol_back_n', 'vol_back_p', 'vol_bat_1', 
        'vol_bat_2', 'vol_block', 'vol_cabt', 'vol_fan', 'vol_gyro_1', 'vol_gyro_2', 'vol_outpwr', 
        'vol_res_n', 'vol_res_p', 'vol_srvo', 'yaw_rate'
    ]
    
    # Run the feature selection process
    best_features, best_r2 = recursive_feature_selection(df, available_features, target_column)
    print(f"\nBest feature set: {best_features} with R²: {best_r2:.3f}")
    
    # Save the best feature set to a file for later use
    save_best_feature_set(df, best_features)

if __name__ == "__main__":
    main()
