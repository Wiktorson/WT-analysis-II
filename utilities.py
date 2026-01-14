import pandas as pd

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from scipy.interpolate import interp1d

#from moepy import lowess
from scipy.optimize import curve_fit # for power curve fitting
import sys
# appending a path with my modules
sys.path.append(r'W:\PD-Engines\Engine Integration Validation\Department\Data Analytics\Wiktor\Python')
import Helpers

sns.set_theme()

def load_data_2024():
    """ Retruns all 3 files in separate df + additional 4th df + column_translation dict
    4th df is all dfs combined together transfered to long format, with adjusted column names"""
    
    
    #df_2022 = pd.read_csv('data\SidebySide_0_0_2022_10_01.csv', delimiter=';', skiprows=[0], low_memory=False) # Old file with missing data
    df_2022 = pd.read_csv(r'data\2024\SidebySidev03_0_0_2022_10_10.csv', delimiter=';', skiprows=[0], low_memory=False) # new file
    df_2022_part2 = pd.read_csv(r'data\2024\SidebySidev03_0_1_2023_04_21.csv', engine='python', on_bad_lines='warn', delimiter=';', skiprows=[0])
    df_2023 = pd.read_csv(r'data\2024\SidebySidev03_0_2_2023_10_31.csv', delimiter=';', skiprows=[0], low_memory=False)
    df_2024 = pd.read_csv(r'data\2024\SidebySidev03_0_3_2024_05_12.csv', delimiter=';', skiprows=[0], low_memory=False)
    for df in [df_2022, df_2022_part2, df_2023, df_2024]:
        df.rename(columns=lambda x: x.replace(' [W. Europe Standard Time]', '').strip(), inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H.%M.%S')
        # There are some columns with WTG9 for some reason, in this analysis I will not look into comparison with that one
        df.drop(columns=[x for x in df.columns if 'WTG009' in x], inplace=True)

    df = pd.concat([df_2022, df_2022_part2, df_2023, df_2024])
    WTG007_columns = [x for x in df.columns if '007' in x]
    WTG008_columns = [x for x in df.columns if '008' in x]

    # Leave only core column names and create separete DFs per WTG7 and WTG8
    df_WTG007 = df[np.hstack(['Timestamp', WTG007_columns])].rename(columns = lambda x: x.split('-')[-1])
    df_WTG007['WT'] = 'WTG007'

    df_WTG008 = df[np.hstack(['Timestamp', WTG008_columns])].rename(columns = lambda x: x.split('-')[-1])
    df_WTG008['WT'] = 'WTG008'

    # Concat into one dataframe with new column definining WTG7 and WTG8 
    df_combined = pd.concat([df_WTG007, df_WTG008])
    # Fixed ProducedMwh column where insetad of . sometimes , was used and covert to float
    df_combined['ProducedMWh (Average MWh)'] = df_combined['ProducedMWh (Average MWh)'].str.replace(',','.').astype('float64')
    
    columns_translation = {x : x.split(' (')[0] for x in df_combined.columns}
    df_combined = df_combined.rename(columns=columns_translation).reset_index(drop=True)
    # switch keys with values for later easier reference
    columns_translation = {val: key for key, val in columns_translation.items()}

    # Make a split between old and new system on WTG7
    df_combined.loc[(df_combined['WT'] == 'WTG007') & (df_combined['Timestamp'] < pd.to_datetime("2023-10-01")), 'WT'] = 'WTG007_old'
    df_combined.loc[(df_combined['WT'] == 'WTG007') & (df_combined['Timestamp'] > pd.to_datetime("2023-10-01")), 'WT'] = 'WTG007_new' # We'll leave one month transition period as not sure when exaclty blades were changed
    # Leave all the values from the transision periods
    df_combined = df_combined[df_combined['WT'].isin(['WTG007_old', 'WTG007_new', 'WTG008'])]
    return [df_2022, df_2023, df_2024, df_combined, columns_translation]

def clean_data(df, filtering_curve):
    print(f"Length before cleaning: {len(df)}")
    # Remove instances where wind speed is higher than 5 (cut in speed) and power is lower than 1
    df_clean = df.copy()
    df_clean = initial_filter_over_1dcurve(df_clean, filtering_curve)
    print(f"Length after cleaning: {len(df_clean)}")
    return df_clean

def fit_logistic_power_curve(df_in):
    """ Retruns parameters of optimal logistic ruve (L, k, x0) and coefficient of determination R2"""
    df = df_in.copy()
    wind_speed_uni, power_uni = get_uniform_windspeed_distribution(df)
    x0_guess = 7 # based on observation of the P(windSpeed)
    k0_guess = 1
    
    # Initial guesses for the parameters L, k, and x0
    initial_guess = [max(power_uni), k0_guess, x0_guess]

    # drop nans for fitting 
    non_nan_ids = (~wind_speed_uni.isna()) & (~power_uni.isna())
    wind_speed_uni = wind_speed_uni[non_nan_ids]
    power_uni = power_uni[non_nan_ids]

    bounds_L_k_x0_min = [power_uni.quantile(0.94), 0, 6] 
    bounds_L_k_x0_max = [power_uni.max(), 1, 10]
    
    params, params_covariance = curve_fit(logistic, wind_speed_uni, power_uni, p0=initial_guess,
                                           bounds=(bounds_L_k_x0_min, bounds_L_k_x0_max))
    
    # Extract the fitted parameters
    L, k, x0 = params
    # drop nans for R2
    non_nan_ids_all = (~df.WindSpeed.isna()) & (~df.ActivePower.isna())
    wind_speed_R2 = df.WindSpeed[non_nan_ids_all].values
    power_R2 = df.ActivePower[non_nan_ids_all].values
    R2 = r2_score(power_R2, logistic(wind_speed_R2, L, k, x0))
    
    # Return the fitted parameters
    return [L, k, x0,], R2

def fit_lowess_power_curve(df_in):
    """ Returns lowess model and R2 fit"""
    df = df_in.copy()
    wind_speed_uni, power_uni = get_uniform_windspeed_distribution(df)

    non_nan_ids = (~wind_speed_uni.isna()) & (~power_uni.isna())
    wind_speed_uni = wind_speed_uni[non_nan_ids]
    power_uni = power_uni[non_nan_ids]
    lowess_model = lowess.Lowess()
    lowess_model.fit(wind_speed_uni.values, power_uni.values, frac=0.2, num_fits=100)

    # drop nans for R2
    non_nan_ids_all = (~df.WindSpeed.isna()) & (~df.ActivePower.isna())
    wind_speed_R2 = df.WindSpeed[non_nan_ids_all].values
    power_R2 = df.ActivePower[non_nan_ids_all].values
    R2 = r2_score(power_R2, lowess_model.predict(wind_speed_R2))
    return lowess_model, R2
     
def get_uniform_windspeed_distribution(df_in, samples_per_bin=200):
    # Make sure we take equal samples per wind speed!
    # We are not interested in fitting the curve that will be influenced by the wind speed distribution!
    df = df_in.copy()
    df['WindSpeedBin'] = df.WindSpeed//1 * 1
    df_uniform = []
    for wind_bin in np.arange(2, 22):
        samples= df[df['WindSpeedBin']==wind_bin][['WindSpeed', 'ActivePower']].sample(samples_per_bin)
        df_uniform.append(samples)
    df_uniform = pd.concat(df_uniform).reset_index(drop=True)
    return df_uniform.WindSpeed, df_uniform.ActivePower

def logistic(x, L, k, x0):
        """
        	L   =	maksymalna wartość krzywej
            k   =	współczynnik wzrostu logistycznego lub nachylenie krzywej
            x0  =	wartość x punktu środkowego krzywej sigmoidalnej
            x   =	liczba rzeczywista
        """
        return L / (1 + np.exp(-k * (x - x0)))

def initial_filter_over_1dcurve(df_in, x_col='WindSpeed', y_col='ActivePower', filtering_curve=None):
    """ Accepts df and returns filtered df - discard all the points that fall below safe power curve  """
    #picked manually looking at scatter power curve, it's safe curve, no good point should be below that
    if filtering_curve == None:
       filtering_curve = calculate_filter_1dcurve()
    predictions = filtering_curve(df_in[x_col].values)
    df = df_in[df_in[y_col] >= predictions ].copy()
    return df

def fit_power_curve_poly6(df):
    wind_speed_uni, power_uni = get_uniform_windspeed_distribution(df)
    # drop nans for fitting 
    non_nan_ids = (~wind_speed_uni.isna()) & (~power_uni.isna())
    wind_speed_uni = wind_speed_uni[non_nan_ids]
    power_uni = power_uni[non_nan_ids]
    poly6_model = np.polyfit(wind_speed_uni, power_uni, 6)

    #pack into poly1d object for convenient
    poly6_model = np.poly1d(poly6_model)
    # drop nans for R2
    non_nan_ids_all = (~df.WindSpeed.isna()) & (~df.ActivePower.isna())
    wind_speed_R2 = df.WindSpeed[non_nan_ids_all].values
    power_R2 = df.ActivePower[non_nan_ids_all].values
    R2 = r2_score(power_R2, poly6_model(wind_speed_R2))

    return poly6_model, R2

def fit_power_curve_IEC(df, bin_size=0.5):
    # Create wind speed bins
    bins = np.arange(0, df.WindSpeed.max() + bin_size, bin_size)
    windSpeedBin = pd.cut(df.WindSpeed, bins=bins, right=False)

    # Group by wind speed bin and calculate average power output
    results = df.ActivePower.groupby(windSpeedBin).mean().reset_index()
    results.columns = ['windSpeedBin', 'averagePowerOutput']

    # Create an interpolation function
    IEC_fit = interp1d(results['windSpeedBin'].apply(lambda x: x.left).values, results['averagePowerOutput'].values, 
                                        bounds_error=False)
    # drop nans for R2
    non_nan_ids_all = (~df.WindSpeed.isna()) & (~df.ActivePower.isna())
    wind_speed_R2 = df.WindSpeed[non_nan_ids_all].values
    power_R2 = df.ActivePower[non_nan_ids_all].values

    pred = IEC_fit(wind_speed_R2)
    non_nan_pred = ~np.isnan(pred)

    R2 = r2_score(power_R2[non_nan_pred], pred[non_nan_pred])
    return IEC_fit, R2

def calculate_filter_1dcurve(max_curve_height = 2000):
    x_points = [0, 6, 10,   15,                 24,                 24.5,   50] 
    y_points = [0, 0, 500,  max_curve_height,   max_curve_height,   0,       0]
    filtering_curve = interp1d(x_points, y_points)
    return filtering_curve

def calculate_filter_1dcurve_hp(max_curve_height = 2900):
    x_points = [0, 6, 10,   16.5,                50,                50,       50] 
    y_points = [0, 0, 500,  max_curve_height,   max_curve_height,   0,       0]
    filtering_curve = interp1d(x_points, y_points)
    return filtering_curve

def fit_linear_regression(df_wide_in, y_col, x_col='ActivePower_WTG008'):
    df_wide_out = df_wide_in.copy()
    df_wide_out = df_wide_out.dropna(subset=[x_col, y_col])

    x = df_wide_out[x_col].to_numpy()[:, np.newaxis]
    y = df_wide_out[y_col].to_numpy()
    linear_relation = LinearRegression(fit_intercept=False).fit(x, y)
    R2 = r2_score(y, linear_relation.predict(x))

    return linear_relation, R2