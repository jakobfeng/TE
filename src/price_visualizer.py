import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta
from matplotlib.dates import DateFormatter

# INPUT VAVLUEs ----------------------------------
data_source = "..\\data\\no2.csv"
from_date = "2019-01-01"
to_date = "2019-01-30"
resolutions = ["hour", "average_day", "w_average_day"]
resol = [0, 1]
save_source_data = str("..\\data\\prices_" + from_date + "_" + to_date + ".csv")
save_source_plot = str("..\\plots\\prices_" + from_date + "_" + to_date + ".png")
# DONE INPUT VALUES -------------------------------


df = pd.read_csv(data_source, header=0, decimal=',')  # full dataset
date_name = df.columns[0]  # date name of column
price_name = df.columns[1]  # price name of column
df[date_name] = pd.to_datetime(df[date_name], dayfirst=False)  # date column to datetime


# visualization

def get_df_between(from_d, to_d):
    from_date = dt.strptime(from_d, "%Y-%m-%d")
    to_date = dt.strptime(to_d, '%Y-%m-%d') + timedelta(hours=23)
    global df
    mask = (df[date_name] > from_date) & (df[date_name] <= to_date)
    data = df.loc[mask]
    data.to_csv(save_source_data, header=True, sep=",", index=False)
    print("Data from " + from_d + " to " + to_d + " saved to file as '" + save_source_data + "'")
    return data


sub_df = get_df_between(from_date, to_date)  # sub data frame hourly resolution


def get_daily_average():
    daily_data = sub_df.groupby(pd.Grouper(key=date_name, freq='D')).mean().dropna()
    daily_data = daily_data.reset_index()
    return daily_data


# plot data based on resolution input
legend_text = ""
if 0 in resol:
    plt.plot(sub_df[date_name], sub_df[price_name], "b-", label="Hourly prices")
if 1 in resol:
    daily_df = get_daily_average()
    plt.plot(daily_df[date_name], daily_df[price_name], color="red", linewidth=5, label="Daily prices")


ax = plt.gca()
ax.set_title("NO2 prices from " + from_date + " to " + to_date, pad=20, size=10)
leg = ax.legend(frameon=False, loc='upper left', ncol=len(resol))
ax.set(xlabel="Date",
       ylabel="EUR/MWh")
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
plt.savefig(save_source_plot)
plt.show()
