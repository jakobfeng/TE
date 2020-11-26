# class for visualizing columns from the training set
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from pathlib import Path

# Markets used to seperate plots into groups
sub_markets = ["NO", "SE", "FI", "DK", "Nordic", "EE", "LV", "LT", "Baltic"]
nordic_markets = ["NO", "SE", "FI", "DK"]
baltic_markets = ["EE", "LV", "LT"]
hydro_markets = ["NO", "SE", "FI"]
nordic_baltic_market = ["Nordic", "Baltic"]
# Data sources possible to plot
data_options = ["Consume", "Hydro", "Price", "Prod", "Sell Vol", "Buy Vol", "Tot Vol"]


def plot(data, sub_markets, resolution, period, save, title):
    path = get_path(resolution)
    columns = get_columns_data(data, sub_markets, resolution)
    df = pd.read_csv(path, usecols=columns)
    df = convert_date_to_datetime(df, resolution)
    df = filter_df_by_period(df, period)
    make_plot(df, resolution, save, data, title)

def plot_double(data, resolution, period, save, title):
    path = get_path(resolution)
    columns = get_columns_data_double(data, resolution)
    df = pd.read_csv(path, usecols=columns)
    df = convert_date_to_datetime(df, resolution)
    df = filter_df_by_period(df, period)
    make_double_plot(df, resolution, save, data, title)


def get_columns_data_double(data, resolution):
    incl_columns = ["Date"]
    if resolution == "h":
        included_cols.append("Hour")
    for data_source in data:
        if data_source == "Price":
            incl_columns.append("System Price")
        elif data_source == "Tot Vol":
            incl_columns.append("Total Vol")
        elif data_source == "Prod":
            incl_columns.extend(["Total Prod"])
        elif data_source == "Consume":
            incl_columns.extend(["Total Consume"])
        elif data_source == "Hydro":
            incl_columns.extend(["Total Hydro"])
    return incl_columns


# Helping method for getting path
def get_path(resolution):
    if resolution == "h":
        path = "..\\data\\input\\combined\\all_data_hourly.csv"
    elif resolution == "d":
        path = "..\\data\\input\\combined\\all_data_daily.csv"
    return path

# Helping method for filtering df on period
def filter_df_by_period(df, period):
    df = df[(df['Date'] >= np.datetime64(period[0])) & (df['Date'] <= np.datetime64(period[1]))]
    return df


# Helping method for making date-column type datetime
def convert_date_to_datetime(df_, resolution):
    date_format = "%d-%m-%Y"
    print(df_.head(1))
    try:
        df_['Date'] = pd.to_datetime(df_['Date'], format=date_format)
    except:
        date_format = "%Y-%m-%d"
        df_['Date'] = pd.to_datetime(df_['Date'], format=date_format)
    if resolution == "h":
        df_["Date"] = pd.to_datetime(df_["Date"]) + pd.to_timedelta(df_["Hour"], unit='h')
        del df_['Hour']
    return df_


# Helping method for making the actual plot
def make_plot(df, resolution, save, data, title):
    columns = df.columns.tolist()
    labels = get_labels_from_columns(df.columns.tolist(), individual=True)
    l_width = get_line_width(data, resolution)
    fig, ax = plt.subplots(figsize=(10, 5.5)) # REMOVE ?
    for i in range(1, len(columns)):
        plt.plot(df[columns[0]], df[columns[i]], linewidth=l_width, label=labels[i])
    for line in plt.legend(loc='upper center', ncol=min(4, len(columns) - 1), bbox_to_anchor=(0.5, 1.05),
                           fancybox=True, shadow=True).get_lines():
        line.set_linewidth(2)
    plt.title(title, pad=20, size=12)
    y_max = max([val for val in df.max().tolist() if (type(val) == np.int64 or type(val) == np.float64)])
    y_min = min([val for val in df.max().tolist() if (type(val) == np.int64 or type(val) == np.float64)])
    axes = plt.gca()
    axes.set_ylim([0, y_max * 1.1])
    axes.set_ylabel(get_y_label(columns[1]), labelpad=6)
    plt.tight_layout()
    if save:
        out_path = get_out_path(data, title)
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()


def make_double_plot(df, resolution, save, data, title):
    columns = df.columns.tolist()
    labels = get_labels_from_columns(df.columns.tolist(), individual=False)
    fig, ax1 = plt.subplots()
    first_color = "tomato"  # price = tomato, other = goldenrod
    second_color = "royalblue"     # when using price, use "royalblue". Else. orchid
    ax1.set_ylabel(get_y_label(columns[1]), labelpad=6, color=first_color)
    ax1.plot(df["Date"], df[columns[1]], color=first_color, label=labels[0])
    #y_max_1 = df[columns[1]].max()
    #ax1.set_ylim([0, y_max_1 * 1.1])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(get_y_label(columns[2]), labelpad=6, color=second_color)
    ax2.plot(df["Date"], df[columns[2]], color = second_color, label=labels[1])
    #y_max_2 = df[columns[2]].max()
    #ax2.set_ylim([0, y_max_2 * 1.1])
    line1, label1 = ax1.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    for line in ax2.legend(line1 + line2, label1 + label2, loc='upper center', ncol=min(4, len(columns) - 1),
               bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True).get_lines():
            line.set_linewidth(2)
    plt.title(title, pad=20, size=12)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        out_path = get_out_path(data, title)
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()



# Helping method for getting y label on axis
def get_y_label(column):
    if "Price" in column:
        return "Euro \u20ac"
    elif "Hydro" in column:
        return "Reservoir GWh"
    else:
        return "MWh"

# Helping method for getting line width
def get_line_width(data, resolution):
    l_width = 0.3 if resolution == "h" else 1
    if "Hydro" in data:
        l_width = 1
    return l_width


# Helping method for getting out path
def get_out_path(data, title):
    if len(data)==1:
        directory = "..\\plots\\data\\{}".format(data[0])
    else:
        directory = "..\\plots\\data\\double"
    existing_paths = sorted(Path(directory).iterdir())
    version = 1
    for p in existing_paths:
        if title in str(p):
            version += 1
    path = directory + "\\{}_v{}.png".format(title, version)
    return path


# Helping method for getting labels
def get_labels_from_columns(columns, individual):
    if individual:
        labels = ["Date"]
        for col in columns:
            for c in sub_markets:
                if c in col:
                    labels.append(c)
            if col == "System Price":
                labels.append("SYS")
    else:
        labels = []
        replace = {"Prod": "Total Production", "Vol": "Total Volume", "Consume": "Total Consumption", "Hydro": "Acc. Hydro State", "System Price": "SYS"}
        for col in columns:
            for key, value in replace.items():
                if key in col:
                    labels.append(value)
    return labels


# Helping method for getting correct columns from all_data set
def get_columns_data(data, sub_markets, resolution):
    included_cols = ["Date"]
    if resolution == "h":
        included_cols.append("Hour")
    for data_source in data:
        if data_source == "Price":
            included_cols.append("System Price")
        elif data_source == "Hydro":
            for m in sub_markets:
                if m in hydro_markets:
                    included_cols.append(m + " Hydro")
        else:
            for m in sub_markets:
                included_cols.append(m + " " + data_source)

    return included_cols


# Helping method for creating plot title for single plot
def create_plot_title(data, markets, period, resolution):
    title = append_markets_title(data, markets)
    title = append_data_title(title, data, resolution)
    title = append_year_title(title, period)
    return title


# Helping method for adding submarket part single plot
def append_markets_title(data, markets):
    if len(data) == 2:
        return ""
    replace_dict_countries = {"NO": "Norway", "SE": "Sweden", "FI": "Finland", "DK": "Denmark", "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania"}
    if len(markets) ==1:  # plotting for only one country/market
        title = market[0]
        for key, value in replace_dict.items():
            title = title.replace(key, value)
    elif markets == ["NO", "SE", "FI", "DK"] or markets == hydro_markets:
        title = "Nordic"
    elif markets == ["EE", "LV", "LT"]:
        title = "Baltic"
    elif markets == ["Nordic", "Baltic"]:
        title = "Nordic vs. Baltic"
    else:
        title = ""
    return title


# Helping method for adding data part to title
def append_data_title(title, data, resolution):
    replace_dict = {"Consume": "Consumption", "Hydro": "Hydro State", "Prod": "Production", "Vol": "Volume",
                        "Tot": "Total"}
    if title == "":
        title += " vs. ".join(data)
    else:
        title += " " + data[0]
    for key, value in replace_dict.items():
        title = title.replace(key, value)
    # Change title if Price is plotted Alone
    if data[0] == "Price" and len(data) ==1 and resolution == "d":
        title = "Daily System Price"
    elif data[0] == "Price" and len(data) ==1 and resolution == "h":
        title = "Hourly System Price"
    return title



# Helping method for adding year part to title
def append_year_title(title, period):
    first_year = str(period[0].year)
    last_year = str(period[1].year)
    if first_year != last_year:
        title += ", {} - {}".format(first_year, last_year)
    else:
        first_month = period[0].strftime("%b")
        last_month = period[1].strftime("%b")
        title += ", {} {} - {} {}".format(first_month, first_year[2:], last_month, last_year[2:])
    return title


if __name__ == '__main__':
    # Data Options: Consume=0, Hydro=1, Price=2, Prod=3, Sell Vol=4, Buy Vol=5, Tot Vol=6
    # Sub Markets: sub_markets, nordic_markets, baltic_markets, nordic_baltic_markets, hydro_markets
    # --------------------------------------------------------------------------------------
    data_options_idx = [2]  # choose. If two are chosen, its a double plot. 6 should not be plottet alone.
    sub_markets_ = nordic_markets # choose
    start_date = datetime.date(2019, 1, 1)  # chose
    end_date = datetime.date(2019, 12, 31)  # chose
    resolution_ = "h"  # choose
    save_ = True  # choose
    # --------------------------------------------------------------------------------------
    data_ = [data_options[i] for i in data_options_idx]
    period_ = [start_date, end_date]
    title_ = create_plot_title(data_, sub_markets_, period_, resolution_)
    if len(data_) == 1:
        plot(data_, sub_markets_, resolution_, period_, save_, title_)
    elif len(data_) == 2:
        plot_double(data_, resolution_, period_, save_, title_)
