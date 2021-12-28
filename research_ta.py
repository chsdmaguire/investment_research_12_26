import pandas as pd
import sqlalchemy
import numpy as np
import ta
from ta import add_all_ta_features

engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:chris32c@localhost:5432/flibyrd_db')

ticker = "'AAPL'"
df = pd.read_sql("select distinct * from stock_prices.candlestick_data where ticker = {} order by date asc".format(ticker), engine)

df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
df['avg_price'] = (df.high + df.low) / 2

df1 = df[['ticker', 'avg_price', 'volume', 'volume_adi',
'volume_em',
'volume_fi',
'volume_mfi',
'volatility_bbp',
'volatility_kcp',
'volatility_kchi',
'volatility_dcp',
'trend_macd_diff',
'trend_adx',
'trend_aroon_ind',
'momentum_rsi',
'momentum_stoch_rsi',
'momentum_ao',
]]

for index, row in df1.iterrows():
    if index > 0:
        df1.loc[index, 'adi_percent'] = (df1.loc[index, 'volume_adi'] - df1.loc[index -1, 'volume_adi']) / df1.loc[index -1, 'volume_adi']

# for index, row in df1.iterrows():
#     if df1.loc[index, 'trend_macd_diff'] > 0 and df1.loc[index -1, 'trend_macd_diff'] < 0:
#         df['norm_macd'] = 0
#     elif df1.loc[index, 'trend_macd_diff'] < 0 and df1.loc[index -1, 'trend_macd_diff'] > 0:
#         df['norm_macd'] = 1
#     else:
#         df1['norm_macd'] = 2

# for index, row in df1.iterrows():
#     if df1.loc[index, 'momentum_ao'] > 0 and df1.loc[index -1, 'momentum_ao'] < 0:
#         df['norm_ao'] = 0
#     elif df1.loc[index, 'momentum_ao'] < 0 and df1.loc[index -1, 'momentum_ao'] > 0:
#         df['norm_ao'] = 1
#     else:
#         df1['norm_ao'] = 2

df1['fi_percentile'] = df1['volume_fi'].rank(pct = True)

for index, row in df1.iterrows():
    if index > 30:
        df1.loc[index, 'forward_20_return'] = (df1.loc[index, 'avg_price'] - df1.loc[index -30, 'avg_price']) / df1.loc[index -30, 'avg_price']

df1['return_percentile'] = df1['forward_20_return'].rank(pct = True)

# for index, row in df1.iterrows():
def buyLabels(df1):
    # if df1['return_percentile'] <= .3333333:
    #     return 0
    # elif df1['return_percentile'] <= .66666666666:
    #     return 1
    # else:
    #     return 2
    if df1['return_percentile'] <= .2:
        return .2
    elif df1['return_percentile'] <= .4 and  df1['return_percentile'] > .2:
        return .4
    elif df1['return_percentile'] <= .6 and  df1['return_percentile'] > .4:
        return .6
    elif df1['return_percentile'] <= .8 and  df1['return_percentile'] > .6:
        return .8
    elif df1['return_percentile'] <= 1 and  df1['return_percentile'] > .8:
        return 1

df1['fi_cat'] = df1.apply(buyLabels, axis=1)

df1 = df1.drop(columns=['avg_price', 'volume', 'momentum_ao', 'trend_macd_diff', 'volume_adi', 'volume_fi', 'return_percentile', 'ticker', 'forward_20_return'])
df1 = df1.dropna()

# times = sorted(df1.index.values)
# last_5pct = sorted(df1.index.values)[-int(0.05*len(times))]

# validation_df = df1[(df1.index  >= last_5pct)]
# train_df = df1[df1.index < last_5pct]

from sklearn import preprocessing
# import random

# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.float64)

# train_df = clean_dataset(train_df)
# validation_df = clean_dataset(validation_df)

for col in df1.columns:
    if col != 'fi_cat':
        df1[col] = df1[col].rank(pct = True)
#         validation_df[col] = preprocessing.scale(validation_df[col].values)

# validation_df = validation_df.dropna()
# train_df = train_df.dropna()

# train_x = train_df
# train_y = train_df
# validation_x = validation_df
# validation_y = validation_df

from keras.models import Sequential
from keras.layers import Dense
train_arr = df1.to_numpy()
print(df1)
X = train_arr[:,0:12]
y = train_arr[:,12]

model = Sequential()
model.add(Dense(24, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

predictions = (model.predict(X) > 0.5).astype(int)

for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))