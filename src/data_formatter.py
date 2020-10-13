import pandas as pd
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import numpy


# get dataframe on hydro weekly from year
def get_hydro_year_weekly(year):
    path = "..\\data\\input\\hydro"
    paths = sorted(Path(path).iterdir())  # list all structured datasets paths
    year_path = ""
    for p in paths:
        if str(year) in str(p):
            year_path = p
    df_year = pd.read_csv(year_path, skiprows=3, usecols=[0, 1, 2])  # exclude Finland
    return df_year


# combine hydro input to one csv file
def write_hydro_all_weekly():
    path = "..\\data\\input\\hydro"
    paths = sorted(Path(path).iterdir())  # list all structured datasets paths
    df_collection = []
    for p in paths:
        df_year = pd.read_csv(p, skiprows=3, usecols=[0, 1, 2])  # exclude Finland
        df_year.rename({'Unnamed: 0': 'Week'}, axis=1, inplace=True)
        df_collection.append(df_year)
    df_all = pd.concat(df_collection, ignore_index=True)
    df_all.to_csv("..\\data\\input\\combined\\hydro_all_weekly.csv", index=False)


# combine hydro input to one csv file
def write_hydro_all_daily():
    path = "..\\data\\input\\combined\\hydro_all_weekly.csv"
    df_year = pd.read_csv(path)  # exclude Finland
    df_daily = pd.DataFrame(columns=['Date', 'NO', 'SE'])
    dates = pd.date_range(start="2014-01-01", end="2019-12-31")
    for date in dates:
        df_daily = df_daily.append({'Date': date}, ignore_index=True)
    for w in df_year.iterrows():
        week_year = w[1][1]
        week = "W" + str(int(week_year[:2]) - 1)
        year = str(2000 + int(week_year[-2:]))
        first_date = datetime.strptime(year + " " + week + " w1", "%Y W%W w%w").date()
        week_dates = [first_date + timedelta(days=x) for x in range(7)]
        no_val = w[1][2]
        se_val = w[1][3]
        for date in week_dates:
            date = numpy.datetime64(date)
            after_start_date = df_daily["Date"] >= date
            before_end_date = df_daily["Date"] <= date
            between_two_dates = after_start_date & before_end_date
            filtered_date = df_daily.loc[between_two_dates]
            if len(filtered_date) == 1:
                index = filtered_date.index[0]
                df_daily.loc[index, "NO"] = no_val
                df_daily.loc[index, "SE"] = se_val
    df_daily.to_csv("..\\data\\input\\combined\\hydro_all_daily.csv", index=False)


def interpolate_hydro():  # just copies. Can be made more sophistacted later
    path = "..\\data\\input\\combined\\hydro_all_daily.csv"
    df = pd.read_csv(path)
    df = df.fillna(method='bfill')
    df = df.fillna(method='pad')
    df.to_csv(path, index=False)


def write_hydro_hourly():
    path = "..\\data\\input\\combined\\hydro_all_daily.csv"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    rng = pd.date_range(df.index.min(), df.index.max() + pd.Timedelta(23, 'H'), freq='H')
    df = df.reindex(rng, method='ffill')
    df.to_csv("..\\data\\input\\combined\\hydro_all_hourly.csv", index_label="Datetime")


def write_price_daily_to_combined():
    path = "..\\data\\input\\price_daily"
    paths = sorted(Path(path).iterdir())  # list all structured datasets paths
    df_years = []
    for p in paths:
        df = pd.read_csv(p, skiprows=2, usecols=[0, 1, 13])  # date, sys, Tr.Heim
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    df_all.rename({'Unnamed: 0': 'Date'}, axis=1, inplace=True)
    df_all.to_csv("..\\data\\input\\combined\\price_all_daily.csv", index=False)


def write_price_hourly_to_combined():
    path = "..\\data\\input\\price_hourly"
    paths = sorted(Path(path).iterdir())  # list all structured datasets paths
    df_years = []
    for p in paths:
        df = pd.read_csv(p, skiprows=2, usecols=[0, 1, 2, 14])  # date, time, sys, Tr.Heim
        df['Hours'] = df['Hours'].str[:2]
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    df_all.rename({'Unnamed: 0': 'Date'}, axis=1, inplace=True)
    df_all['Hours'] = pd.to_datetime(df_all['Hours'], format='%H').dt.time
    df_all['Date'] = pd.to_datetime(df_all['Date'], format='%d-%m-%Y')
    df_all.to_csv("..\\data\\input\\combined\\price_all_hourly.csv", index=False)


if __name__ == '__main__':
    # df = get_hydro_year_weekly(2015)
    # print(df.head())
    # write_hydro_all_weekly()
    # write_hydro_all_daily()
    # interpolate_hydro()
    # write_hydro_hourly()
    # write_price_daily_to_combined()
     write_price_hourly_to_combined()
