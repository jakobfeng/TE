import pandas as pd

phase_3_model = pd.read_csv("data/model_analysis.csv")
phase_3_metrics = pd.read_csv("data/metrics_analysis.csv")


def analyse_model():
    df = phase_3_model[["SID", 'Title', 'Models', 'Approach', 'Method', 'Training / test size']]
    model_id = {"1": [], "3": [], "4": [], "5": [], "Hybrid": []}
    replace = {"1": "Multi-Agent", "3": "Reduced form", "4": "Statistical", "5": "CI", "Hybrid": "Hybrid"}
    model_type = {"1": [], "3": [], "4": [], "5": [], "Hybrid": []}
    for index, row in df.iterrows():
        id = row["SID"]
        model_cat = row['Approach']
        model = row['Method']
        if len(model_cat) > 1:
            model_id["Hybrid"].append(id)
            model_type["Hybrid"].append(model)
        else:
            model_id[model_cat].append(id)
            model_type[model_cat].append(model)
    print("Overview:")
    for key, value in model_id.items():
        print("{} has {} studies".format(replace[key], len(value)))
    print("\nModels:")
    model_count = {}
    for key, model_list in model_type.items():
        for model in model_list:
            m_list = model.split(", ")
            for m in m_list:
                m = m.lower()
                if m not in model_count.keys():
                    model_count[m] = 1
                else:
                    model_count[m] += 1
    res_df = pd.DataFrame.from_dict(model_count, orient="index")
    print(res_df)


def analyse_metrics():
    metrics_dict = {}
    for index, row in phase_3_metrics.iterrows():
        id = row["SID"][1:]
        metric_list = row["Performance metric(s) used"].split(", ")
        for m in metric_list:
            if m not in metrics_dict.keys():
                metrics_dict[m] = [id]
            else:
                metrics_dict[m].append(id)
    metric_df = pd.DataFrame(columns=["Name", "Explination", "Count", "ID"])
    for key, value in metrics_dict.items():
        val_string = ", ".join(value)
        row_dict = {"Name": key, "Explination": "", "Count": len(value), "ID": val_string}
        metric_df = metric_df.append(row_dict, ignore_index=True)
    #metric_df.to_csv("data\\metric_summary.csv", index=False)
    point_forecasts = ["MAE", "RMSE", "MSE", "MAPE", "SMAPE", "R^2"]
    prob_forecasts = ["CRPS", "BS", "PL", "ES", "ACE", "PICP", "UC", "CC", "RI", "SC", "PINAW", "AWD", "WS", "CWC"]
    count = 0
    point_count = 0
    prob_count = 0
    for index, row in phase_3_metrics.iterrows():
        count += 1
        point = False
        prob = False
        metric_list = row["Performance metric(s) used"].split(", ")
        for m in metric_list:
            if m in point_forecasts:
                point = True
            if m in prob_forecasts:
                prob = True
        if point:
            point_count += 1
        if prob:
            prob_count += 1
    print("Number of studies with point metrics: {} of {}, equalling {}%".format(point_count, count, round(100*point_count/count, 1)))
    print("Number of studies with prob metrics: {} of {}, equalling {}%".format(prob_count, count, round(100*prob_count/count, 1)))

if __name__ == '__main__':
    analyse_metrics()
