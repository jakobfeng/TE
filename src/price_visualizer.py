import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from src.price_df_provider import Price_df_provider

# INPUT VALUES ----------------------------------
data_source = "..\\data\\no2.csv"
from_date = "2019-01-01"
to_date = "2019-01-14"
resolutions = ["hour", "average_day", "w_average_day"]
resol = [0, 1]
# DONE INPUT VALUES -------------------------------

data = Price_df_provider(data_source, from_date, to_date, ",")  # get price data object
df = data.get_data_frame()  # get price dataframe

save_source_data = str("..\\data\\prices_" + from_date + "_" + to_date + ".csv")
save_source_plot = str("..\\plots\\prices_" + from_date + "_" + to_date + ".png")
date_name = df.columns[0]  # date name of column
price_name = df.columns[1]  # price name of column

# plot data based on resolution input
legend_text = ""
if 0 in resol:
    plt.plot(df[date_name], df[price_name], "b-", label="Hourly prices")
if 1 in resol:
    daily_df = data.get_daily_average_duplicate_first_day()
    print(daily_df)

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
