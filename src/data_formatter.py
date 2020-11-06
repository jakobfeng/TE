from distutils.command.config import config
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import math

# helping method reformatting hour column
def reformat_hour_column(df, date_sep):
    df['Hour'] = pd.to_datetime(df['Hour'], format='%H').dt.time
    format = "%d{}%m{}%Y".format(date_sep, date_sep)
    df['Date'] = pd.to_datetime(df['Date'], format=format)
    return df


def rename_column_names(df, col_name_changes):
    for from_name, to_name in col_name_changes.items():
        df.rename({from_name: to_name}, axis=1, inplace=True)
    return df


def convert_folder_to_csv(paths, replace_commas, header_row, resolution, make_integer):
    for p in paths:
        p_list = str(p).split("\\")
        del p_list[-2] #remove raw from the list
        p_list[-1] = p_list[-1][:-3] + "csv"
        out_path = "\\".join(p_list)
        df = pd.read_html(str(p), thousands=None, header=header_row, encoding="utf-8")[0]
        if replace_commas:
            #df = df.apply(lambda x: x.str.replace(',', '.'))
            for y in df.columns:
                if (df[y].dtype == object):
                    df[y] = df[y].apply(lambda x: str(x).replace(',', '.'))
        if resolution == "h":
            col_name_changes = {'Unnamed: 0': 'Date', 'Hours': 'Hour'}
            int_columns_from = 2
        elif resolution == "d":
            col_name_changes = {'Unnamed: 0': 'Date'}
            int_columns_from = 1
        df = rename_column_names(df, col_name_changes)
        if make_integer:
            df = make_all_columns_integers_without_date(df, int_columns_from)
        df.to_csv(out_path, sep=",", index=False, encoding="utf-8")
        print("Saved " + out_path + " to file")


def get_all_csv_files_from_directory(path):
    result = []
    paths = sorted(Path(path).iterdir())  # list all files in directory
    for p in paths:
        if ".csv" in str(p):
            result.append(p)
    return result


def write_price_to_combined(resolution, convert_to_csv, replace_commas):
    if resolution =="h":
        path ="..\\data\\input\\price_hourly"
        out_path = "..\\data\\input\\combined\\price_hourly.csv"
        columns = [0, 1, 2] # date=0, hour=1, sys=2, (Tr.Heim=14)
    elif resolution =="d":
        path = "..\\data\\input\\price_daily"
        out_path = "..\\data\\input\\combined\\price_daily.csv"
        columns = [0, 1] # date=0, sys=1, (Tr.Heim=13)
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p , usecols=columns)
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    if resolution =="h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
    col_name_changes = {'SYS': 'System Price'}
    df_all = rename_column_names(df_all, col_name_changes)
    df_all.to_csv(out_path, index=False)

# helping method
def aggregate_markets_volume(df):
    market_to_columns = {
        'NO Buy Vol': ["NO1 Buy", "NO2 Buy","NO3 Buy", "NO4 Buy","NO5 Buy"],
        'NO Sell Vol': ["NO1 Sell", "NO2 Sell", "NO3 Sell", "NO4 Sell", "NO5 Sell"],
        'SE Buy Vol': ["SE1 Buy", "SE2 Buy", "SE3 Buy", "SE4 Buy"],
        'SE Sell Vol': ["SE1 Sell", "SE2 Sell", "SE3 Sell", "SE4 Sell"],
        'FI Buy Vol': ["FI Buy"],
        'FI Sell Vol': ["FI Sell"],
        'DK Buy Vol': ["DK1 Buy", "DK2 Buy"],
        'DK Sell Vol': ["DK1 Sell", "DK2 Sell"],
        "Nordic Buy Vol": ["NO Buy Vol", "SE Buy Vol", "FI Buy Vol", "DK Buy Vol"],
        "Nordic Sell Vol": ["NO Sell Vol", "SE Sell Vol", "FI Sell Vol", "DK Sell Vol"],
        'Baltic Buy Vol': ["EE Buy", "LV Buy", "LT Buy"],
        'Baltic Sell Vol': ["EE Sell", "LV Sell" ,"LT Sell"]
    }
    for market in market_to_columns.keys():
        df[market] = df[market_to_columns[market]].sum(axis=1)
    return df

def write_volume_to_combined(resolution, convert_to_csv, replace_commas):
    if resolution =="h":
        path ="..\\data\\input\\volume_hourly"
        out_path = "..\\data\\input\\combined\\volume_hourly.csv"
    elif resolution =="d":
        path = "..\\data\\input\\volume_daily"
        out_path = "..\\data\\input\\combined\\volume_daily.csv"
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p, sep=",")
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    if resolution == "h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
    col_name_changes = {'Turnover at system price': 'Total Vol'}
    df_all = rename_column_names(df_all, col_name_changes)
    df_all = aggregate_markets_volume(df_all)
    incl_columns = ["Date", "Total Vol", "NO Buy Vol", "NO Sell Vol","SE Buy Vol", "SE Sell Vol","DK Buy Vol",
                    "DK Sell Vol", "FI Buy Vol", "FI Sell Vol","Nordic Buy Vol", "Nordic Sell Vol","Baltic Buy Vol", "Baltic Sell Vol"]
    if resolution == "h":
        incl_columns.insert(1, "Hour")
    df_all = df_all[incl_columns]
    df_all.to_csv(out_path, index=False, float_format='%g')

def fix_error_in_year_hydro(df):
    numer_of_weeks = df.shape[0]
    for i in range(0, numer_of_weeks-1):
        week = df.iloc[i,0]
        next_week = df.iloc[i+1,0]
        year = int(week[-2:])
        next_year = int(next_week[-2:])
        difference = next_year - year
        if difference == 2:
            correct_year = year+1
            correct_string = next_week[:-2] + str(correct_year)
            df.iloc[i+1, df.columns.get_loc('Date')] = correct_string
    return df

# combine weekly hydro input to one csv file
def write_hydro_all_weekly(convert_to_csv, replace_commas):
    path = "..\\data\\input\\hydro_weekly"
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        if "hydro_weekly.csv" not in str(p):
            df_year = pd.read_csv(p, sep=",")
            df_years.append(df_year)
    df_all = pd.concat(df_years, ignore_index=True)
    col_name_changes = {"NO": "NO Hydro", "SE": "SE Hydro","FI": "FI Hydro"}
    df_all = rename_column_names(df_all, col_name_changes)
    df_all.drop(df_all.tail(2).index, inplace=True) # drop last 2 weeks of 2020, no data there
    df_all = df_all.interpolate(method='linear', axis=0).ffill().bfill() # affects only two last rows of 2019
    df_all = df_all[50:365] #from the last week of 2013 to the second week of 2020
    df_all = fix_error_in_year_hydro(df_all)
    df_all.to_csv("..\\data\\input\\hydro_weekly\\hydro_weekly.csv", index=False, float_format='%g')

def interpolate_week(first, last, week_dates):
    values = np.linspace(first,last,8).round(2)
    date_values = {}
    for i in range(len(week_dates)):
        date_values[np.datetime64(week_dates[i])] = values[i]
    return date_values

# combine hydro input to one csv file
def write_hydro_daily_to_combined():
    path = "..\\data\\input\\hydro_weekly\\hydro_weekly.csv"
    df_weekly = pd.read_csv(path)
    df_daily = pd.DataFrame(columns=['Date', 'NO Hydro', 'SE Hydro', 'FI Hydro'])
    dates = pd.date_range(start="2014-01-01", end="2019-12-31")
    for date in dates:
        df_daily = df_daily.append({'Date': date}, ignore_index=True)
    number_of_weeks = df_weekly.shape[0]
    for i in range(0, number_of_weeks-1):
        week_row = df_weekly.iloc[i, :]
        next_week_row = df_weekly.iloc[i+1, :]
        week_year = week_row[0]
        week = "W" + str(int(week_year[:2]))
        year = str(2000 + int(week_year[-2:]))
        minus_one_years = ["2014", "2015", "2019", "2020"] #  Se hydro weekly data. Confusion with weeks. Some years starts at 0, some at 1.
        if year in minus_one_years:
            week = "W" + str(int(week_year[:2])-1)  # minus one. Weeks starts at 0 these years
        first_date = datetime.strptime(year + " " + week + " w1", "%Y W%W w%w").date()
        print("Year {}. W{}, firstdate {}, minus 1: {}".format(year, week_year[:2], first_date, year in minus_one_years))
        week_dates = [first_date + timedelta(days=x) for x in range(7)]
        no_date_values = interpolate_week(week_row[1], next_week_row[1], week_dates)
        se_date_values = interpolate_week(week_row[2], next_week_row[2], week_dates)
        fi_date_values = interpolate_week(week_row[3], next_week_row[3], week_dates)
        for date in week_dates:
            date = np.datetime64(date)
            after_start_date = df_daily["Date"] >= date
            before_end_date = df_daily["Date"] <= date
            between_two_dates = after_start_date & before_end_date
            filtered_date = df_daily.loc[between_two_dates]
            if len(filtered_date) == 1:
                index = filtered_date.index[0]
                df_daily.loc[index, "NO Hydro"] = no_date_values[date]
                df_daily.loc[index, "SE Hydro"] = se_date_values[date]
                df_daily.loc[index, "FI Hydro"] = fi_date_values[date]
    df_daily["Date"] = df_daily["Date"].dt.strftime("%d-%m-%Y")
    df_daily["Total Hydro"] = df_daily["NO Hydro"] + df_daily["SE Hydro"] + df_daily["FI Hydro"]
    df_daily["Total Hydro"] = pd.to_numeric(df_daily["Total Hydro"])
    df_daily.to_csv("..\\data\\input\\combined\\hydro_daily.csv", index=False, float_format='%g')


#helping method for hydro hourly
def append_23_more_rows(df):
    last_row = df.tail(1)
    date = last_row.iloc[0, 0]
    no = last_row.iloc[0, 2]
    se = last_row.iloc[0, 3]
    fi = last_row.iloc[0, 4]
    tot = last_row.iloc[0, 5]
    for i in range(1,24):
        row = {"Date": date, "Hour": i, "NO Hydro": no, "SE Hydro": se, "FI Hydro": fi, "Total Hydro": tot}
        df = df.append(row, ignore_index=True)
    return df


def write_hydro_hourly_to_combined():
    path = "..\\data\\input\\combined\\hydro_daily.csv"
    df_daily = pd.read_csv(path)
    df_daily['Date'] = pd.to_datetime(df_daily['Date'], format="%d-%m-%Y")
    df_hourly = df_daily.set_index('Date').resample('H').interpolate('linear')
    df_hourly['Hour'] = [d.time() for d in df_hourly.index]
    df_hourly['Date'] = [d.date() for d in df_hourly.index]
    df_hourly["Date"] = pd.to_datetime(df_hourly["Date"], format="%Y-%m-%d")
    df_hourly["Date"] = df_hourly["Date"].dt.strftime("%d-%m-%Y")
    df_hourly['Hour'] = df_hourly["Hour"].apply(lambda x: x.hour)
    df_hourly = df_hourly[["Date", "Hour", "NO Hydro", "SE Hydro", "FI Hydro", "Total Hydro"]]
    df_hourly = append_23_more_rows(df_hourly)
    df_hourly.to_csv("..\\data\\input\\combined\\hydro_hourly.csv", index=False, float_format='%g')

def plot_hydro(resolution):
    path = ""
    save_source_plot = ""
    if resolution =="w":
        path = path = "..\\data\\input\\hydro_weekly\\hydro_weekly.csv"
        save_source_plot = str("..\\plots\\hydro_week.png")
        title = "Hydro Week"
    elif resolution =="d":
        path = path = "..\\data\\input\\combined\\hydro_daily.csv"
        save_source_plot = str("..\\plots\\hydro_day.png")
        title = "Hydro Day"
    elif resolution =="h":
        path = path = "..\\data\\input\\combined\\hydro_hourly.csv"
        save_source_plot = str("..\\plots\\hydro_hour.png")
        title = "Hydro Hour"
    df = pd.read_csv(path)
    plt.plot(df["NO Hydro"], label="NO")
    plt.plot(df["SE Hydro"], label="SE")
    plt.plot(df["FI Hydro"], label="FI")
    ax = plt.gca()
    ax.set_title(title, pad=20, size=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    ax.set(xlabel="Index",
           ylabel="Reservoir GWh")
    plt.savefig(save_source_plot)
    plt.close()


def make_all_columns_integers_without_date(df, int_columns):
    cols = df.columns
    df[cols[int_columns:]] = df[cols[int_columns:]].apply(pd.to_numeric, errors='coerce')
    return df

def write_consumption_to_combined(resolution, convert_to_csv, replace_commas):
    if resolution =="h":
        path ="..\\data\\input\\consumption_hourly"
        out_path = "..\\data\\input\\combined\\consumption_hourly.csv"
    elif resolution =="d":
        path = "..\\data\\input\\consumption_daily"
        out_path = "..\\data\\input\\combined\\consumption_daily.csv"
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2, resolution = resolution, make_integer=True)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p, sep=",")
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    if resolution == "h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
        df_all = copy_last_days_hours_in_column(df_all, "EE")  # affects last 3 rows of 2019, column EE
        df_all['Baltic'] = df_all['EE'] + df_all['LV'] + df_all["LT"]
        df_all = copy_last_days_hours_in_column(df_all, "SE")  # affects only last 13 rows of 2019, column SE
        df_all['Nordic'] = df_all['NO'] + df_all['SE'] + df_all["FI"] + df_all["DK"]
    col_name_changes = {'NO': 'NO Consume', "SE": "SE Consume", "FI": "FI Consume", "DK": "DK Consume",
                        "Nordic": "Nordic Consume", "EE": "EE Consume", "LV": "LV Consume", "LT": "LT Consume",
                        "Baltic": "Baltic Consume"}
    df_all = rename_column_names(df_all, col_name_changes)
    df_all.to_csv(out_path, index=False, float_format='%g')


def copy_last_days_hours_in_column(df, column):
    for i in range(len(df)-1,1,-1):
        if i >23:
            value = df[column].iloc[i]
            if pd.isna(value):
                new_value = df[column].iloc[i-24]
                df.at[i, column] = new_value
    return df


def write_production_to_combined(resolution, convert_to_csv, replace_commas):
    if resolution =="h":
        path ="..\\data\\input\\production_hourly"
        out_path = "..\\data\\input\\combined\\production_hourly.csv"
    elif resolution =="d":
        path = "..\\data\\input\\production_daily"
        out_path = "..\\data\\input\\combined\\production_daily.csv"
    if convert_to_csv:
        raw_path = path + "\\raw"
        raw_paths = sorted(Path(raw_path).iterdir())  # list all raw xls files
        convert_folder_to_csv(raw_paths, replace_commas, header_row=2, resolution=resolution, make_integer=True)
    paths = get_all_csv_files_from_directory(path)
    df_years = []
    for p in paths:
        df = pd.read_csv(p, sep=",")
        df_years.append(df)
    df_all = pd.concat(df_years, ignore_index=True)
    if resolution == "h":
        df_all['Hour'] = df_all['Hour'].str[:2].astype(int)
        df_all = copy_last_days_hours_in_column(df_all, "EE")  # affects only last 16 rows of 2019, column EE
        df_all['Baltic'] = df_all['EE'] + df_all['LV'] + df_all["LT"]
    col_name_changes = {'NO': 'NO Prod', "SE": "SE Prod", "FI": "FI Prod", "DK": "DK Prod",
                        "Nordic": "Nordic Prod", "EE": "EE Prod", "LV": "LV Prod", "LT": "LT Prod",
                        "Baltic": "Baltic Prod"}
    df_all = rename_column_names(df_all, col_name_changes)
    df_all["Total Prod"] = df_all["Nordic Prod"] + df_all["Baltic Prod"]
    df_all.to_csv(out_path, index=False, float_format='%g')


def combine_all_data(resolution):
    path = "..\\data\\input\\combined"
    all_paths = sorted(Path(path).iterdir())  # list all datasets paths
    paths = []
    for p in all_paths:
        if (resolution in str(p)) and  ("all_data_" not in str(p)):
            paths.append(p)
    dfs = []
    for p in paths:
        df = pd.read_csv(p,sep=",")
        dfs.append(df)
    for i in range(len(dfs)-2,-1,-1):
        print("Merging in {}".format(paths[i]))
        if resolution =="daily":
            df = pd.merge(df, dfs[i], on='Date', how='outer')
        elif resolution=="hourly":
            df = pd.merge(df, dfs[i], on=['Date', 'Hour'], how='outer')
    if resolution == "daily":
        ordered_columns = ['Date', "System Price"]
        all_columns = df.columns.tolist()
        other_columns = [col for col in all_columns if col not in ordered_columns]
        ordered_columns.extend(other_columns)
        df = df[ordered_columns]
        df.to_csv("..\\data\\input\\combined\\all_data_daily.csv", index = False)
    elif resolution == "hourly":
        df.dropna() #  remove all rows with nan (hydro dataset has not accounted for summer time)
        ordered_columns = ['Date', "Hour", "System Price"]
        all_columns = df.columns.tolist()
        other_columns = [col for col in all_columns if col not in ordered_columns]
        ordered_columns.extend(other_columns)
        df = df[ordered_columns]
        date_format = "%d-%m-%Y"
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        df.to_csv("..\\data\\input\\combined\\all_data_hourly.csv", index = False, float_format='%g')
    else:
        print("HAS TO BE HOURLY OG DAILY")
        assert False

def add_time_columns_to_all_data(resolution):
    if resolution == "d":
        data_path = "..\\data\\input\\combined\\all_data_daily.csv"
    else:
        data_path = "..\\data\\input\\combined\\all_data_hourly.csv"
    date_format = "%Y-%m-%d"
    df = pd.read_csv(data_path, sep=",")
    df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    for index, row in df.iterrows():
        date = row[0]
        week = datetime.date(date).isocalendar()[1]
        sine_week = math.sin(math.pi*week/52)
        df.loc[index, "Sine Week"] = round(sine_week, 3)
        month = date.month
        sine_month = math.sin(math.pi*month/12)
        df.loc[index, "Sine Month"] = round(sine_month, 3)
        season = math.floor(date.month/4) + 1
        sine_season = math.sin(math.pi*season/4)
        df.loc[index, "Sine Season"] = round(sine_season, 3)
    out_path = data_path
    #out_path  = "..\\data\\input\\combined\\all_data_daily_weeks.csv"
    df.to_csv(out_path, sep=",", index=False, float_format='%g')


def add_hydro_deviations_to_all_data(resolution):
    if resolution == "d":
        data_path = "..\\data\\input\\combined\\all_data_daily.csv"
    else:
        data_path = "..\\data\\input\\combined\\all_data_hourly.csv"
    df = pd.read_csv(data_path, sep=",")
    date_format = "%Y-%m-%d"
    df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    hydro_df = df[["Date", "NO Hydro", "SE Hydro","FI Hydro","Total Hydro"]]
    average_year_df = hydro_df.groupby([hydro_df["Date"].dt.month.rename("Month"), hydro_df["Date"].dt.day.rename("Day")]).mean()
    column_rename_dict = {"NO Hydro": "NO Mean", "SE Hydro": "SE Mean", "FI Hydro": "FI Mean", "Total Hydro": "Total Mean"}
    average_year_df = rename_column_names(mean_hydro_df, column_rename_dict)
    country_names = ["NO", "SE", "FI", "Total"]
    for index, row in df.iterrows():
        day = row["Date"].day
        month = row["Date"].month
        for country in country_names:
            mean_col_name = country+" Mean"
            day_row = average_year_df.loc[month, day]
            mean = day_row[mean_col_name]
            hydro = row[country + " Hydro"]
            dev = hydro - mean
            df.loc[index, country + " Hydro Dev"] = round(dev, 3)
    out_path = data_path
    df.to_csv(out_path, sep=",", index=False, float_format='%g')


if __name__ == '__main__':
    print("Running method.." + "\n")
    #write_price_to_combined("d", convert_to_csv=True, replace_commas=True)
    #write_price_to_combined("h", convert_to_csv=True, replace_commas=True)
    # write_volume_to_combined("d", convert_to_csv=False, replace_commas=True)
    # write_volume_to_combined("h", convert_to_csv=False, replace_commas=True)
    # write_hydro_all_weekly(convert_to_csv = False, replace_commas=False) # replace_commas=False, always
    # write_hydro_daily_to_combined()
    # write_hydro_hourly_to_combined()
    # write_consumption_to_combined("d", convert_to_csv=False, replace_commas=True)
    # write_consumption_to_combined("h", convert_to_csv=False, replace_commas=True)
    # write_production_to_combined("d", convert_to_csv=False, replace_commas=True)
    # write_production_to_combined("h", convert_to_csv=False, replace_commas=True)
    # combine_all_data("daily")
    # combine_all_data("hourly")
    # add_time_columns_to_all_data("d")
    # add_time_columns_to_all_data("h")
    # add_hydro_deviations_to_all_data("d")
    # add_hydro_deviations_to_all_data("h")


