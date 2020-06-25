#!pip install pytrends

from datetime import date, timedelta
from functools import partial
from time import sleep
from calendar import monthrange

import pandas as pd

from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq

def get_last_date_of_month(year: int, month: int) -> date:
    """Given a year and a month returns an instance of the date class
    containing the last day of the corresponding month.
    Source: https://stackoverflow.com/questions/42950/get-last-day-of-the-month-in-python
    """
    return date(year, month, monthrange(year, month)[1])


def convert_dates_to_timeframe(start: date, stop: date) -> str:
    """Given two dates, returns a stringified version of the interval between
    the two dates which is used to retrieve data for a specific time frame
    from Google Trends.
    """
    return f"{start.strftime('%Y-%m-%d')} {stop.strftime('%Y-%m-%d')}"


def _fetch_data(pytrends, build_payload, timeframe: str) -> pd.DataFrame:
    """Attempts to fecth data and retries in case of a ResponseError."""
    attempts, fetched = 0, False
    while not fetched:
        try:
            build_payload(timeframe=timeframe)
        except ResponseError as err:
            print(err)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break
        else:
            fetched = True
    return pytrends.interest_over_time()


def get_daily_data(word: str,
                 start_year: int,
                 start_mon: int,
                 stop_year: int,
                 stop_mon: int,
                 geo: str = 'US',
                 verbose: bool = True,
                 wait_time: float = 5.0) -> pd.DataFrame:
    """Given a word, fetches daily search volume data from Google Trends and
    returns results in a pandas DataFrame.
    Details: Due to the way Google Trends scales and returns data, special
    care needs to be taken to make the daily data comparable over different
    months. To do that, we download daily data on a month by month basis,
    and also monthly data. The monthly data is downloaded in one go, so that
    the monthly values are comparable amongst themselves and can be used to
    scale the daily data. The daily data is scaled by multiplying the daily
    value by the monthly search volume divided by 100.
    For a more detailed explanation see http://bit.ly/trendsscaling
    Args:
        word (str): Word to fetch daily data for.
        start_year (int): the start year
        start_mon (int): start 1st day of the month
        stop_year (int): the end year
        stop_mon (int): end at the last day of the month
        geo (str): geolocation
        verbose (bool): If True, then prints the word and current time frame
            we are fecthing the data for.
    Returns:
        complete (pd.DataFrame): Contains 4 columns.
            The column named after the word argument contains the daily search
            volume already scaled and comparable through time.
            The column f'{word}_unscaled' is the original daily data fetched
            month by month, and it is not comparable across different months
            (but is comparable within a month).
            The column f'{word}_monthly' contains the original monthly data
            fetched at once. The values in this column have been backfilled
            so that there are no NaN present.
            The column 'scale' contains the scale used to obtain the scaled
            daily data.
    """

    # Set up start and stop dates
    start_date = date(start_year, start_mon, 1)
    stop_date = get_last_date_of_month(stop_year, stop_mon)

    # Start pytrends for US region
    pytrends = TrendReq(hl='en-US', tz=360)
    # Initialize build_payload with the word we need data for
    build_payload = partial(pytrends.build_payload,
                            kw_list=[word], cat=0, geo=geo, gprop='')

    # Obtain monthly data for all months in years [start_year, stop_year]
    monthly = _fetch_data(pytrends, build_payload,
                         convert_dates_to_timeframe(start_date, stop_date))

    # Get daily data, month by month
    results = {}
    # if a timeout or too many requests error occur we need to adjust wait time
    current = start_date
    while current < stop_date:
        last_date_of_month = get_last_date_of_month(current.year, current.month)
        timeframe = convert_dates_to_timeframe(current, last_date_of_month)
        if verbose:
            print(f'{word}:{timeframe}')
        results[current] = _fetch_data(pytrends, build_payload, timeframe)
        current = last_date_of_month + timedelta(days=1)
        sleep(wait_time)  # don't go too fast or Google will send 429s

    daily = pd.concat(results.values()).drop(columns=['isPartial'])
    complete = daily.join(monthly, lsuffix='_unscaled', rsuffix='_monthly')

    # Scale daily data by monthly weights so the data is comparable
    complete[f'{word}_monthly'].ffill(inplace=True)  # fill NaN values
    complete['scale'] = complete[f'{word}_monthly'] / 100
    complete[word] = complete[f'{word}_unscaled'] * complete.scale

    return complete

"""# Model"""

#%tensorflow_version 2.x

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from sklearn import metrics
import math

from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from sklearn import metrics
import math
from sklearn.linear_model import LinearRegression

def preprocess(files):
  df_list = []
  for file in files:
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    close_df = df[['date','close']]
    close_df.dropna(how = 'any',inplace=True)
    df_list.append(close_df)
  
  concat_df = pd.concat(df_list,ignore_index=True)
  concat_df.index = concat_df['date']
  test = concat_df[['close']]
  
  # import new data source - pytrends
  start_year=test.index[0].year
  start_month=test.index[0].month
  stop_year=test.index[-1].year
  stop_month=test.index[-1].month
  pytrends_df = get_daily_data('bitcoin', start_year, start_month, stop_year, stop_month, geo='US')


  # get target pytrends
  date = test.index.to_list()
  subset_pytrends_df = pytrends_df.loc[date]

  # get pytrends_shift_df --> part of X2_scaled
  pytrends_shift = subset_pytrends_df[['bitcoin']]
  for i in range(1,21,1):
      shift = 'bitcoin_before' + str(i)
      pytrends_shift[shift] = pytrends_shift.bitcoin.shift(i)
  pytrends_shift.dropna(inplace=True)
  pytrends_shift.drop(columns=['bitcoin'],inplace=True)
  scaler = MinMaxScaler(feature_range=(0,1))
  X2_scaled = scaler.fit_transform(pytrends_shift)

  # scale training dataset
  scaler = MinMaxScaler(feature_range=(0,1))
  test_scaled = scaler.fit_transform(test)
  joblib.dump(scaler,'scaler.save')

  # spilt test into X and y
  X1_scaled = []
  y_scaled = []
  test_length = 20
  for i in range(test_length,len(test_scaled)):
      X1_scaled.append(test_scaled[i-test_length:i,0])
      y_scaled.append(test_scaled[i,0])
  X1_scaled = np.array(X1_scaled)
  y_scaled = np.array(y_scaled)
  X1_scaled = np.reshape(X1_scaled,(X1_scaled.shape[0],X1_scaled.shape[1],1))
  y_scaled = y_scaled.reshape(-1,1)
  y = scaler.inverse_transform(y_scaled)

  return X1_scaled, X2_scaled, y

def model(model_filename, X1_scaled):
  LSTM_model = Sequential()
  LSTM_model.add(LSTM(10,return_sequences=True,input_shape=(20,1),recurrent_dropout=0.4))
  LSTM_model.add(LSTM(2,recurrent_dropout=0.4))
  LSTM_model.add(Dense(1))
  LSTM_model.compile(optimizer= 'adam', loss= 'MSE')
  LSTM_model.load_weights(model_filename)

  # predict
  y1_hat_scaled = LSTM_model.predict(X1_scaled)
  y1_hat_scaled = y1_hat_scaled.reshape(-1,1)

  return y1_hat_scaled

def combiner(y1_hat_scaled,X2_scaled,combiner_filename,scaler_filename):
  # generate whole X2_scaled
  X2_scaled = np.append(X2_scaled, y1_hat_scaled, axis=1)
  X2_scaled = np.reshape(X2_scaled,(X2_scaled.shape[0],X2_scaled.shape[1],1))
  model = tf.keras.models.load_model(combiner_filename)

  # predict
  y2_hat_scaled = model.predict(X2_scaled)

  # inverse_scale
  scaler = joblib.load(scaler_filename)
  y2_hat = scaler.inverse_transform(y2_hat_scaled)

  return y2_hat


