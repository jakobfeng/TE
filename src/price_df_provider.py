# class for providing other classes with prices as dataframe
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta


class Price_df_provider:
    def __init__(self, data_source, from_date, to_date, decimal_sep):
        df = pd.read_csv(data_source, header=0, decimal=decimal_sep)  # full dataset
        self.date_name = df.columns[0]  # date name of column
        df[self.date_name] = pd.to_datetime(df[self.date_name], dayfirst=False)  # date column to datetime
        self.start_date = dt.strptime(from_date, "%Y-%m-%d")
        self.end_date = dt.strptime(to_date, '%Y-%m-%d') + timedelta(hours=23)
        mask = (df[self.date_name] >= self.start_date) & (df[self.date_name] <= self.end_date)
        self.data = df.loc[mask]
        self.price_name = df.columns[1]  # date name of column

    def get_start_date(self):
        return self.from_date

    def get_end_date(self):
        return self.to_date

    def get_data_frame(self):
        return self.data

    def get_daily_average(self):
        daily_data = self.data.groupby(pd.Grouper(key=self.date_name, freq='D')).mean().dropna()
        self.daily_data = daily_data.reset_index()
        return self.daily_data

    def get_daily_average_duplicate_first_day(self):  #  mostly used for plotting
        daily_df = self.get_daily_average()
        last_day = daily_df[self.date_name].iloc[-1]
        next_day = last_day + timedelta(days=1)
        copy_daily_df = daily_df.copy()
        copy_daily_df = copy_daily_df.append({self.date_name: next_day, self.price_name: None}, ignore_index=True)
        first_price = daily_df[self.price_name][daily_df.index[0]]
        copy_daily_df[self.price_name] = copy_daily_df[self.price_name].shift(1)
        copy_daily_df.at[0, self.price_name] = first_price
        return copy_daily_df


if __name__ == '__main__':
    source = "..\\data\\no2.csv"
    from_date = "2019-01-01"
    to_date = "2019-01-03"
    data = Price_df_provider(source, from_date, to_date, ",")
    df = data.get_data_frame()
    df_daily = data.get_daily_average()
    print(df_daily)
