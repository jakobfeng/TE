# script for presenting results from slr phase 2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

deleted_df = pd.read_csv("data\\deleted_phase_1_618.csv", encoding="utf-8")
phase_2_df = pd.read_csv("data\\phase_2_670.csv", encoding="utf-8")
replace_dict_model = {"1": "Multi-agent", "2": "Fundamental", "3": "Reduced Form", "4": "Statistical",
                      "5": "CI", "6": "Hybrid"}
replace_dict_horizon = {"1": "Short", "2": "Medium", "3": "Long"}
five_model_colors = ["plum", "moccasin", "lightcoral", 'lightskyblue', "mediumaquamarine"]
six_model_colors = ["plum", "moccasin", "lightcoral", 'lightskyblue', "mediumaquamarine", "silver"]


# Helping Method for Pie Plots
def add_white_circle():
    centre_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
    plt.gca().add_artist(centre_circle)


# Helping method for finishing plot
def show_and_save(path):
    plt.tight_layout()
    plt.savefig("plots\\" + path)
    plt.show()
    plt.close()


# Helping methods
def get_fig_size():
    return 9.5, 4.8


def set_plot_size():
    plt.subplots(figsize=get_fig_size())


def plot_saved_and_deleted_articles():
    number_phase_2 = len(phase_2_df)
    number_deleted = len(deleted_df)
    x = [number_phase_2, number_deleted]
    label = ["Kept", "Deleted"]
    total_count = number_phase_2 + number_deleted
    colors = ['lightskyblue', "lightcoral"]
    set_plot_size()
    plt.pie(x, startangle=90, colors=colors, frame=True, textprops=None,
            autopct=lambda p: '{:.0f}\n({:.0f}%)'.format((p / 100) * total_count, p), pctdistance=0.45)
    add_white_circle()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(label, loc="upper right", fancybox=True, shadow=True)
    plt.axis('equal')
    plt.title("Outcome of Initial Manual Filter Stage")
    show_and_save("saved_vs_deleted.png")


def plot_model_type_distribution():
    df_grouped = phase_2_df.groupby(by=["Type of method/model"])
    hybrids = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    methods = {"Hybrid": 0}
    horizon = {"Hybrid": {'1': 0, '2': 0, '3': 0, 'Mix': 0}, "Multi-agent": None, "Fundamental": None,
               "Reduced Form": None, "Statistical": None, "CI": None}
    for index, row in df_grouped:
        if len(index) > 1:
            category = "Hybrid"
            methods[category] += len(row)
            for i in str.replace(index, ",", "").split():
                hybrids[i] += len(row)
            hor_dict = get_horizon_dict(row)
            for key, value in hor_dict.items():
                horizon["Hybrid"][key] += value
        else:
            category = replace_dict_model[index]
            methods[category] = len(row)
            horizon[category] = get_horizon_dict(row)
    print("Methods: {} ".format(methods))
    print("Sum of methods: {}".format(sum([v for v in methods.values()])))
    print("Hybrid distribution: {}".format(hybrids))
    print("Sum of hybrids: {}".format(sum([v for v in hybrids.values()]) / 2))
    print("Horizons: {} ".format(horizon))
    horizon_sums = {k: sum([v for v in horizon[k].values()]) for k in horizon.keys()}
    print("Sum of horizon: {}".format(sum([v for v in horizon_sums.values()])))
    df = pd.DataFrame(horizon).transpose()
    df = df.reindex(["Hybrid", "CI", "Statistical", "Reduced Form", "Fundamental", "Multi-agent"])
    df = df.fillna(0)
    df["2"] = df["2"] + df["Mix"]  # remove mix, add them to mid-term
    df = df[["1", "2", "3"]]
    print("Method per horizon no mix")
    print(df)
    colors_ = ["mediumaquamarine", 'lightskyblue', "lightcoral"]
    title = "Classification of Studies"
    df.plot(kind="barh", color=colors_, stacked=True, title=title, figsize=get_fig_size(), width=0.75)
    plt.legend(["Short Term", "Med. Term", "Long Term"], fancybox=True, shadow=True)
    plt.xlabel("Number of studies")
    plt.ylabel("Model approach")
    plt.tick_params(bottom=True, left=False, labelleft=True, labelbottom=True)
    values = [int(a) for a in df.sum(axis=1).tolist()]
    plt.xlim(0, max(values)*1.08)
    print("Distribution per model: {}".format(values))
    for i, v in enumerate(values):
        txt = "{}%".format(round((v/sum(values))*100, 1))
        plt.text(v+15, i, txt, color="black", size=10, ha='center')
    show_and_save("methods_and_horizon.png")


# Helping method
def get_horizon_dict(row):
    horizons = row["Horizon"].tolist()
    d = {x: horizons.count(x) for x in horizons}
    mix_hor_keys = [k for k in d.keys() if len(k) > 1]
    if len(mix_hor_keys) > 0:
        mix_horizon_count = sum([v for k, v in d.items() if k in mix_hor_keys])
        for key in mix_hor_keys:
            del d[key]
        d["Mix"] = mix_horizon_count
    return d


def plot_hybrid_distribution():
    count = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    df_grouped = phase_2_df.groupby(by=["Type of method/model"])
    for index, row in df_grouped:
        if len(index) > 1:
            for key in count.keys():
                if key in index:
                    count[key] += len(row)
    keys = [k for k in count.keys()]
    names = [replace_dict_model[k] for k in keys]
    values = [count[k] for k in keys]
    set_plot_size()
    plt.bar(names, values, color=five_model_colors)
    plt.title("Hybrid Approach Distribution")
    plt.ylabel("Number of studies")
    plt.xlabel("Model approach")
    for i, v in enumerate(values):
        plt.text(i, v + 1, str(v), color="black", size=8, ha='center')
    plt.ylim(0, max(values) * 1.1)
    plt.tick_params(bottom=False)
    show_and_save("hybrid_distribution.png")


def plot_probabilistic_pie():
    print("\nPlotting probabilistic pie..")
    desired_output = ["1, 2", "2"]
    prob_df = phase_2_df[phase_2_df["Output"].isin(desired_output)]
    prob_count = {"1": 0, "2": 0, "4": 0, "3": 0, "5": 0, "6": 0}
    for index, row in prob_df.groupby(by=["Type of method/model"]):
        if len(index) > 1:
            prob_count["6"] += len(row)
        else:
            prob_count[index] += len(row)
    number_ids = [k for k, v in prob_count.items()]
    values = [prob_count[k] for k in number_ids]
    names = [replace_dict_model[k] for k in number_ids]
    print(values)
    print(names)
    colors_ = six_model_colors
    colors_[2], colors_[3] = colors_[3], colors_[2]
    set_plot_size()
    plt.pie(values, startangle=90, frame=True, textprops=None, colors=colors_,
            autopct=lambda p: '{:.0f}'.format((p / 100) * sum(values), p) if p > 0 else "", pctdistance=0.53)
    add_white_circle()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(names, loc="upper right", fancybox=True, shadow=True)
    plt.axis('equal')
    plt.title("Probabilistic Forecast per Model Approach")
    show_and_save("probabilistic_output.png")


def plot_probabilistic_per_horizon():
    print("\nPlotting probabilistic per horizon..")
    desired_output = ["1, 2", "2"]
    prob_df = phase_2_df[phase_2_df["Output"].isin(desired_output)]
    prob_count = {"1": [0, 0], "2": [0, 0], "3": [0, 0]}
    prob_proportion = {"1": None, "2": None, "3": None}
    prob_count = count_horizon_rows(0, phase_2_df, prob_count)
    prob_count = count_horizon_rows(1, prob_df, prob_count)
    print(prob_count)
    for k, v in prob_count.items():
        print("{}\tPoint: {}, Prob: {}".format(replace_dict_horizon[k], v[0] - v[1], v[1]))
    for horizon in prob_count.keys():
        prob = prob_count[horizon][1] / prob_count[horizon][0]
        point = 1 - prob
        prob_proportion[horizon] = [point, prob]
    print(prob_proportion)
    prob_proportion_df = pd.DataFrame(prob_proportion).transpose()
    colors = ["lightcoral", 'lightskyblue']
    title = "Forecast Type per Horizon"
    prob_proportion_df.plot(kind="bar", stacked=True, color=colors, title=title, figsize=get_fig_size())
    plt.legend(["Point", "Prob."], fancybox=True, shadow=True, ncol=2)
    plt.xticks(range(len(replace_dict_horizon)), [v for v in replace_dict_horizon.values()], rotation=0)
    plt.tick_params(bottom=False)
    for i, v in enumerate([v[1] for v in prob_count.values()]):
        percent = int(prob_proportion[str(i + 1)][1] * 100)
        bar_string = "{}% ({})".format(percent, v)
        plt.text(i, 1.02, bar_string, color="black", size=8, ha='center')
    plt.ylabel("Proportion")
    plt.xlabel("Prediction horizon")
    plt.ylim(0, 1.2)
    show_and_save("prediction_type_per_horizon.png")


# Helping method
def count_horizon_rows(position, df, prob_count):
    for index, row in df.groupby(by=["Horizon"]):
        if "2" in index:
            prob_count["2"][position] += len(row)
        elif "3" in index:
            prob_count["3"][position] += len(row)
        else:
            prob_count["1"][position] += len(row)
    return prob_count


def plot_ci_per_year():
    print("\nPlotting ci articles per year..")
    ci_df = pd.read_csv("data\\ci_methods_not_classified.csv")
    print("Total number of CI: " + str(len(ci_df)))
    ci_year_df = ci_df.groupby(by=["Year", "Document Type"]).size().reset_index(name="Count")
    art_df = ci_year_df[ci_year_df["Document Type"]=="Article"]
    conf_df = ci_year_df[ci_year_df["Document Type"]=="Conference"]
    years = [i for i in range(min(art_df["Year"].values[0], conf_df["Year"].values[0]), max(art_df["Year"].values[-1], conf_df["Year"].values[-1])+1)]
    new_df = pd.DataFrame(columns=["Year", "Articles", "Conference Proceedings", "Total"])
    for y in years:
        a_df = art_df[art_df["Year"]==y]
        if len(a_df)==0:
            a = 0
        else:
            a = a_df["Count"].tolist()[0]
        c_df = conf_df[conf_df["Year"]==y]
        if len(c_df)==0:
            c = 0
        else:
            c = c_df["Count"].tolist()[0]
        new_df = new_df.append({"Year": y, "Articles": a, "Conference Proceedings": c, "Total": a+c}, ignore_index=True)
    df_to_plot = new_df[["Year", "Articles", "Conference Proceedings"]]
    df_to_plot = df_to_plot.set_index("Year")
    article_count = new_df["Total"].tolist()
    print(article_count)
    colors = ["seagreen", "mediumaquamarine"]
    ax = plt.gca()
    df_to_plot.plot(ax=ax, kind="bar", stacked=True, color=colors, figsize=get_fig_size())
    plt.title("Computational Intelligent Studies per Year")
    plt.ylabel("Number of studies")
    plt.xlabel("Time")
    article_count_average = [(article_count[0] + article_count[1]) / 2]
    for i in range(1, len(years) - 1):
        prev = article_count[i - 1]
        this = article_count[i]
        next_ = article_count[i + 1]
        avg = (prev + this + next_) / 3
        article_count_average.append(avg)
    article_count_average.append((article_count[-1] + article_count[-2]) / 2)
    new_df["Average"] = article_count_average
    new_df["Average"].plot.line(years, article_count_average, color="seagreen", linewidth=4, label="3 year mean of total studies", ax=ax)
    plt.xticks(rotation=90)
    plt.legend(loc="upper left", fancybox=True, shadow=True)
    show_and_save("ci_articles_time.png")


def plot_model_type_within_ci():
    print("\nPlotting model type within CI articles..")
    df = pd.read_csv("data/ci_methods_classified.csv")
    print("Number of articles: {}".format(len(df)))
    nn_list = ["1", "2", "3"]
    neural_net = {True: 0, False: 0}
    for index, row in df.iterrows():
        nn = False
        for id in nn_list:
            if id in row["Approach"]:
                nn = True
        if nn:
            neural_net[True] += 1
        else:
            neural_net[False] += 1
    print("Neural net in studies: {}".format(neural_net))
    count = {}
    df_grouped = df.groupby(by=["Approach"])
    for category, row in df_grouped:
        cat_list = category.replace(" ", "").split(",")
        for cat in cat_list:
            if cat not in count.keys():
                count[cat] = len(row)
            else:
                count[cat] += len(row)
    print(count)
    print("Sum of counts: {}".format(sum(count.values())))
    replace_keys = {"1": "FFNN", "2": "RNN", "3": "Fuzzy", "4": "SVM", "5": "Bio.", "6": "Ensamble", "7": "Other"}
    green = "mediumaquamarine"
    set_plot_size()
    tuples = sorted(count.items())
    keys = [k[0] for k in tuples]
    values = [count[k] for k in keys]
    keys = [replace_keys[k] for k in keys]
    plt.bar(keys, values, color=green)
    plt.ylim(0, 1.1 * max(count.values()))
    for i, v in enumerate(values):
        plt.text(i, v + 5, str(v), color="black", size=8, ha='center')
    plt.ylabel("Number of studies")
    plt.xlabel("Model type")
    plt.title("Computational Intelligent Model Distribution")
    show_and_save("ci_method_dist.png")


def plot_models_type_within_ffnn():
    print("\nPlotting model type within ffnn..")
    df = pd.read_csv("data/ci_methods_classified.csv")
    print("Number of articles: {}".format(len(df)))
    feed_forward = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        if "1" in row["Approach"]:
            feed_forward = feed_forward.append(row, ignore_index=True)
    ff_grouped = feed_forward.groupby(by=["Model"]).size().reset_index(name="count")
    print(ff_grouped)
    dist = {"a": 0, "b": 0, "c": 0, "d": 0}
    leg_names = ["Multilayer percpetron (MLP)", "Extreme learning machine (ELM)", "Radial basis function (RBF)", "Convolutional neural net (CNN)"]
    for category, row in ff_grouped.iterrows():
        labels = row[0].split(", ")
        count = row[1]
        for l in labels:
            dist[l] += count
    print(dist)
    x = dist.values()
    four_greens = ["mediumaquamarine", "seagreen", "gainsboro", "limegreen"]
    set_plot_size()
    plt.pie(x, colors=four_greens, autopct=lambda p: '{:.0f}%'.format(p), pctdistance=0.5, frame=True)
    add_white_circle()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(leg_names, loc="upper right", fancybox=True, shadow=True)
    plt.title("Feed-Forward Neural Network Distribution")
    plt.axis('equal')
    show_and_save("ffnn_dist.png")


def plot_model_type_within_stat():
    print("\nPlotting model type within statistical articles..")
    df = pd.read_csv("data/stat_methods_classified.csv")
    print("Number of articles: {}".format(len(df)))
    count = {}
    df_grouped = df.groupby(by=["Approach"])
    for category, row in df_grouped:
        cat_list = category.replace(" ", "").split(",")
        for cat in cat_list:
            if cat not in count.keys():
                count[cat] = len(row)
            else:
                count[cat] += len(row)
    print(count)
    print("Sum of counts: {}".format(sum(count.values())))
    replace_keys = {"1": "SD/ES", "2": "Regression", "3": "AR-types", "4": "Grey", "5": "GARCH", "6": "Other"}
    blue = "lightskyblue"
    set_plot_size()
    tuples = sorted(count.items())
    keys = [k[0] for k in tuples]
    values = [count[k] for k in keys]
    keys = [replace_keys[k] for k in keys]
    plt.bar(keys, values, color=blue)
    plt.ylim(0, 1.1 * max(count.values()))
    for i, v in enumerate(values):
        plt.text(i, v + 5, str(v), color="black", size=8, ha='center')
    plt.ylabel("Number of studies")
    plt.xlabel("Model type")
    plt.title("Statistical Model Distribution")
    show_and_save("stat_method_dist.png")
    # ------------------------------------------------------
    reg_count = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}
    reg_names = ["Linear", "Density", "Quantile", "Logistic", "Dynamic"]
    desired_approach = ["2", "1, 2, 3", "1, 2", "2, 3", "2, 4", "2, 5", "2, 6"]
    reg_df = df[df["Approach"].isin(desired_approach)]
    print("Length of regression df: {}".format(len(reg_df)))
    for index, row in reg_df.iterrows():
        model_list = row["Model"].split(", ")
        for m in model_list:
            reg_count[m] += 1
    print(reg_count)
    x = reg_count.values()
    five_greens = ["lightskyblue", "teal", "gainsboro", "deepskyblue", "royalblue"]
    set_plot_size()
    plt.pie(x, startangle=90, colors=five_greens, autopct=lambda p: '{:.0f}%'.format(p), pctdistance=0.5, frame=True)
    add_white_circle()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(reg_names, loc="upper right", fancybox=True, shadow=True)
    plt.title("Statistical Regression Model Distribution")
    plt.axis('equal')
    show_and_save("regression_dist.png")
    # ------------------------------------------------------
    print("\nPlotting distribution within AR-types")
    desired_approach = ["3", "1, 2, 3", "1, 3", "2, 3", "3, 4", "3, 5", "3, 6"]
    ar_df = df[df["Approach"].isin(desired_approach)]
    ar_grouped = ar_df.groupby(by="Name").size().reset_index(name="Count")
    print("Total sum after groping: {}".format(ar_grouped["Count"].sum()))
    count = {"ARIMA": 0, "AR": 0, "ARMA": 0, "ARMAX": 0, "SARIMA": 0, "VAR": 0, "ARX": 0, "SETAR": 0}
    for index, row in ar_df.iterrows():
        app_list = row["Name"].split(", ")
        for app in app_list:
            if app in count.keys():
                count[app] += 1
    print(count)
    print("Total sum after counting: {}".format(sum(count.values())))
    x = count.values()
    names = count.keys()
    eight_blues = ["lightskyblue", "teal", "gainsboro", "deepskyblue", "cadetblue", "navy", "cornflowerblue", "powderblue"]
    set_plot_size()
    plt.pie(x, startangle=90, colors=eight_blues, autopct=lambda p: '{:.0f}%'.format(p), pctdistance=0.5, frame=True)
    add_white_circle()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(names, loc="upper right", fancybox=True, shadow=True)
    plt.title("Statistical AR-type Model Distribution")
    plt.axis('equal')
    show_and_save("artype_dist.png")


def plot_articles_per_year():
    print("\nPlotting number of studies per year..")
    phase_2_year_df = phase_2_df.groupby(by=["Year", "Document Type"]).size().reset_index(name="Count")
    articles = phase_2_year_df[phase_2_year_df["Document Type"] == "Article"]
    conferences = phase_2_year_df[phase_2_year_df["Document Type"] == "Conference"]
    c_years = conferences["Year"].tolist()
    a_years = articles["Year"].tolist()
    years = [y for y in range(min(c_years[0], a_years[0]), max(c_years[-1], a_years[-1])+1)]
    df = pd.DataFrame(columns=["Year", "Articles", "Conference Papers", "Total"])
    for y in years:
        a = articles[articles["Year"] == y]
        if len(a) == 0:
            a = 0
        else:
            a = np.array(a["Count"])[0]
        c = conferences[conferences["Year"] == y]
        if len(c) == 0:
            c = 0
        else:
            c = np.array(c["Count"])[0]
        row = {"Year": y, "Articles": a, "Conference Papers": c, "Total": a+c}
        df = df.append(row, ignore_index=True)
    df_art_conf = df[["Year", "Articles", "Conference Papers"]].set_index("Year")
    print("Number of articles: {}, number of conference papers: {}".format(sum(df_art_conf["Articles"]), sum(df_art_conf["Conference Papers"])))
    df_art_conf.plot(kind="bar", stacked=True, color=["lightcoral", "lightskyblue", ], label=["Articles", "Conference papers"],
                     figsize=get_fig_size())
    plt.title("Studies per Year")
    plt.ylabel("Number of studies")
    plt.xlabel(None)
    plt.legend(loc="upper left", fancybox=True, shadow=True)
    show_and_save("articles_per_year.png")


if __name__ == '__main__':
    print("Running methods..\n")
    plot_saved_and_deleted_articles()
    plot_model_type_distribution()
    plot_hybrid_distribution()
    plot_probabilistic_pie()
    plot_probabilistic_per_horizon()
    plot_ci_per_year()
    plot_model_type_within_ci()
    plot_models_type_within_ffnn()
    plot_model_type_within_stat()
    plot_articles_per_year()
