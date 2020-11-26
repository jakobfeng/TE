# Script for sorting files from the first output of search
import pandas as pd

full_df = pd.read_csv("data\\phase_2_682.csv", header=0, sep=",", encoding="utf-8")


def get_subset(model_type, horizon, market, output_type):
    result_df = full_df[full_df["Type of method/model"].isin(model_type)]
    result_df = result_df[result_df["Horizon"].isin(horizon)]
    result_df = result_df[result_df["Market"].isin(market)]
    result_df = result_df[result_df["Output"].isin(output_type)]
    result_df = result_df.rename(columns={"Type of method/model": "Method"})
    result_df = result_df.reset_index()
    result_df = result_df[["Authors", "Title", "Year", "Cited by", "Method", "Horizon", "Market", "Output"]]
    result_df = short_string_column(result_df, "Title", 8)
    result_df = short_author_list(result_df, "Authors", 2)
    result_df = change_numbers_to_words(result_df)
    result_df = result_df.sort_values(by=["Year"])
    result_df.to_csv("data\\data_hybrid_short_medium_dam_probabilistic_2.csv", index=False)
    return result_df


def change_numbers_to_words(df):
    method_dict = {"1": "MA", "2": "FU", "3": "RF", "4": "ST", "5": "DD"}
    hor_dict = {"1": "S", "2": "M", "3": "L", "1, 2": "S/M"}
    mar_dict = {"1": "DAM", "2": "IDM", "3": "BM"}
    out_dict = {"1": "PO", "2": "DI"}
    for index, row in df.iterrows():
        method = row[4]
        horizon = row[5]
        market = row[6]
        output = row[7]
        if method in method_dict.keys():
            df.at[index, "Method"] = method_dict[method]
        else:
            df.at[index, "Method"] = "HY"
        if horizon in hor_dict.keys():
            df.at[index, "Horizon"] = hor_dict[horizon]
        if market in mar_dict.keys():
            df.at[index, "Market"] = mar_dict[market]
        if output in out_dict.keys():
            df.at[index, "Output"] = out_dict[output]
    return df


def short_string_column(df, col_name, limit):
    f = lambda x: " ".join(x[col_name].split()[0:limit]) + "..." if len(x[col_name].split()) > limit else x[col_name]
    df[col_name] = df.apply(f, axis=1)
    return df


def short_author_list(df, col_name, limit):
    f = lambda x: ", ".join(x[col_name].split(",")[0:limit]) + " et al." if len(x[col_name].split()) > limit else x[col_name]
    df[col_name] = df.apply(f, axis=1)
    return df



def get_data_hybrid_short_medium_dam_probabilistic():
    m_type = ["5", "1, 5", "2, 5", "3, 5", "4, 5"]
    hor = ["1", "2", "1, 2", "3"]
    mar = ["1", "2", "3", "1, 3"]
    out_type = ["2", "1, 2"]
    final_df = get_subset(m_type, hor, mar, out_type)
    return final_df


# Model type: multiagent (1), fundamental (2), reduced form (3), statistical (4), data (5)
# Horizon: short (1), medium (2), long (3)
# Market: DAM (1), IDM (2), BM (3)
# output: point (1), probabilistic (2)
if __name__ == '__main__':
    df = get_data_hybrid_short_medium_dam_probabilistic()
    #print(df.head())
