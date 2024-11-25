from ecmwf import ecmwf
from gfs import gfs
from weather import weather
from gglows_historical import gglows_historical
from gglows_forecast import gglows_forecast
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkan import TKAN
from tcn import TCN

class PinballLoss(tf.keras.losses.Loss):
    def __init__(self, quantile=0.6, reduction=tf.keras.losses.Reduction.AUTO, name='pinball_loss'):
        super().__init__(name=name)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantile * error, (self.quantile - 1) * error))

def prepare_forecast_data(data, target, n_steps):
    # The input sequence is the last `n_steps` rows of data, similar to how X_test is prepared.
    seq_x = data.iloc[-n_steps:, :]  # Last n_steps rows
    seq_x = np.hstack((seq_x, target.iloc[-n_steps:].values.reshape(-1, 1)))

    return seq_x  # Reshape to fit the model's input format


# Setup parameters for open-meteo forecast
latitude = [43.35, 42.83, 43.74, 43.27, 43.80, 43.52, 43.16, 43.16, 42.85, 43.04, 42.60, 42.84, 42.96,
            42.96, 42.73, 44.54, 44.76, 43.26, 44.09, 43.51, 44.44, 43.62, 43.93, 43.95]  # geographic coordinates
longitude = [19.36, 19.52, 19.71, 19.99, 19.30, 18.79, 18.85, 19.12, 19.88, 19.74, 19.94, 20.17, 19.58,
             19.10, 19.79, 19.23, 19.20, 18.61, 18.95, 18.45, 19.15, 19.37, 18.79, 19.57]
meteo_station = ["Pljevlja", "Kolašin", "Zlatibor", "Sjenica", "Višegrad", "Foča", "Plužine", "Žabljak",
                 # station names
                 "Berane", "Bijelo Polje", "Plav", "Rožaje", "Mojkovac", "Šavnik", "Andrijevica", "Loznica",
                 "Bijeljina", "Čemerno", "Han Pijesak", "Kalinovik", "Zvornik", "Rudo", "Sokolac", "Goražde"]

# Define river ids (LINKNO) and names
river_ids = [220252711, 220249952, 220212799, 220227955, 220232074,
             220267840, 220302223, 220284319, 220348963, 220214203]
hydro_stations = ["Uvac", "Kokin Brod", "Bistrica", "Piva", "HS Prijepolje",
                  "Potpeć", "Višegrad", "Bajina Bašta", "Zvornik", "HS Đurđevića Tara"]

past_days = 2  # weather info for how many past days (possible values: 0, 1, 2, 3, 5, 7, 14, 31, 61, 92)
forecast_days = 7  # weather info for how many future days (possible values: 1, 3, 5, 7, 10, 15)

ecmwf(longitude, latitude, meteo_station, past_days, forecast_days)
gfs(longitude, latitude, meteo_station, past_days, forecast_days)
weather(longitude, latitude, meteo_station, past_days, forecast_days)
gglows_historical(river_ids, hydro_stations)

# Get today's date in the required format
today_str = datetime.now().strftime('%Y-%m-%d')

# Define dynamic file names using today's date
precip_hist_file = f'ecmwf_precip_hist_{today_str}.csv'
precip_forecast_file = f'ecmwf_precip_forecast_{today_str}.csv'
temp_hist_file = f'ecmwf_temp_hist_{today_str}.csv'
temp_forecast_file = f'ecmwf_temp_forecast_{today_str}.csv'
flow_file = f'gglows_discharge_{today_str}.csv'

# Load and concatenate precipitation data
a = pd.read_csv(precip_hist_file, parse_dates=[0], index_col=0)
b = pd.read_csv(precip_forecast_file, parse_dates=[0], index_col=0)
prec = pd.concat([a, b])

# Load and concatenate temperature data
a = pd.read_csv(temp_hist_file, parse_dates=[0], index_col=0)
b = pd.read_csv(temp_forecast_file, parse_dates=[0], index_col=0)
temp = pd.concat([a, b])

# Load flow data
flow = pd.read_csv(flow_file, parse_dates=[0], index_col=0)


all_stations = pd.merge(temp, prec, on='date', suffixes=['_temp', '_pad'])

# Define meteo for hydro_stations
meteo_stations_visegrad = ['Plav', 'Andrijevica', 'Berane', 'Rožaje', 'Mojkovac', 'Kolašin', 'Bijelo Polje', 
                           'Sjenica', 'Pljevlja', 'Šavnik', 'Žabljak', 'Plužine', 'Čemerno', 'Kalinovik', 'Foča', 
                           'Goražde','Rudo', 'Višegrad', 'Zlatibor','Sokolac', 'Han Pijesak']
meteo_stations_bbasta = ['Plav', 'Andrijevica', 'Berane', 'Rožaje', 'Mojkovac', 'Kolašin', 'Bijelo Polje', 
                         'Sjenica', 'Pljevlja', 'Šavnik', 'Žabljak', 'Plužine', 'Čemerno', 'Kalinovik', 'Foča', 
                         'Goražde', 'Rudo', 'Višegrad', 'Zlatibor','Sokolac', 'Han Pijesak']
meteo_stations_zvornik = ['Plav', 'Andrijevica', 'Berane', 'Rožaje', 'Mojkovac', 'Kolašin', 'Bijelo Polje', 
                          'Sjenica','Pljevlja', 'Šavnik', 'Žabljak', 'Plužine', 'Čemerno', 'Kalinovik', 'Foča', 
                          'Goražde', 'Rudo', 'Višegrad', 'Zlatibor','Sokolac', 'Han Pijesak', 'Zvornik']
meteo_stations_piva = ['Čemerno', 'Plužine', 'Šavnik', 'Žabljak']
meteo_stations_uvac = ['Sjenica', 'Rožaje', 'Bijelo Polje']
meteo_stations_kbrod = ['Sjenica', 'Rožaje', 'Bijelo Polje', 'Zlatibor']
meteo_stations_bistrica = ['Sjenica', 'Rožaje', 'Bijelo Polje', 'Zlatibor']
meteo_stations_potpec = ['Plav', 'Andrijevica', 'Berane', 'Rožaje', 'Mojkovac', 'Kolašin', 'Bijelo Polje', 
                         'Sjenica', 'Pljevlja', 'Rudo', 'Zlatibor']
meteo_stations_prijepolje = ['Plav', 'Andrijevica', 'Berane', 'Rožaje', 'Mojkovac', 'Kolašin', 'Bijelo Polje', 
                         'Sjenica', 'Pljevlja']
meteo_stations_tara = ['Pljevlja', 'Žabljak', 'Šavnik', 'Kolašin', 'Mojkovac', 'Andrijevica']

pick = {'Višegrad': meteo_stations_visegrad,
        'Bajina Bašta': meteo_stations_bbasta,
        'Zvornik': meteo_stations_zvornik,
        'Piva': meteo_stations_piva,
        'Uvac': meteo_stations_uvac,
        'Kokin Brod': meteo_stations_kbrod,
        'Bistrica': meteo_stations_bistrica,
        'Potpeć': meteo_stations_potpec,
        'HS Prijepolje': meteo_stations_prijepolje,
        'HS Đurđevića Tara': meteo_stations_tara}

# Fill in the hydrological stations later (few models not available)
hydro_stations = ["Uvac", "Kokin Brod", "Piva", "HS Prijepolje", "Zvornik"]

# Go through all the hydro_stations
for hydro_station in hydro_stations:
    meteo_stations = pick[hydro_station]
    specific = [f'{station}_temp' for station in meteo_stations] + [f'{station}_pad' for station in meteo_stations]
    
    # Prepare input data
    X_forecast = prepare_forecast_data(all_stations[specific], flow[hydro_station], 7)
    X_forecast = np.nan_to_num(X_forecast)
    
    folder_path = '' + hydro_station
    
    model_tcn = load_model(folder_path+'_tcn_pinball.keras', custom_objects={'PinballLoss': PinballLoss, 'TCN':TCN})
    print(model_tcn.summary())
    
    model_lstm = load_model(folder_path+'_lstm_pinball.keras', custom_objects={'PinballLoss': PinballLoss})
    print(model_lstm.summary())
    
    model_tkan = load_model(folder_path+'_tkan_pinball.keras', custom_objects={'PinballLoss': PinballLoss})
    print(model_tkan.summary())
    
    X_forecast = np.nan_to_num(X_forecast)
    
    # Generate a date range starting from today
    forecast_dates = [datetime.now().date() + timedelta(days=i) for i in range(5)]
    lstm_forecast = model_lstm.predict(X_forecast[None, :])
    lstm_df = pd.DataFrame({
    'Date': forecast_dates,
    'LSTM_Forecast': lstm_forecast[0]
})
    tcn_forecast = model_tcn.predict(X_forecast[None, :])
    tcn_df = pd.DataFrame({
    'Date': forecast_dates,
    'TCN_Forecast': tcn_forecast[0]
})
    tkan_forecast = model_tkan.predict(X_forecast[None, :])
    tkan_df = pd.DataFrame({
    'Date': forecast_dates,
    'TKAN_Forecast': tkan_forecast[0]
})
    
    # Optionally, save the DataFrames to CSV or display them
    lstm_df.to_csv(f"{hydro_station}_lstm_forecast.csv", index=False)
    tcn_df.to_csv(f"{hydro_station}_tcn_forecast.csv", index=False)
    tkan_df.to_csv(f"{hydro_station}_tkan_forecast.csv", index=False)
