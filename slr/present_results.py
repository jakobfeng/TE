# script for presenting results from slr phase 2
import pandas as pd
import matplotlib.pyplot as plt

deleted_df = pd.read_csv("data\\deleted_phase_1_606.csv", encoding="utf-8")
phase_2_df = pd.read_csv("data\\phase_2_682.csv", encoding="utf-8")
replace_dict_model = {"1": "Multi-Agent", "2": "Fundamental", "3": "Reduced Form", "4": "Statistical",
                      "5": "Data-driven", "6": "Hybrid"}
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
def set_plot_size():
    plt.subplots(figsize=(9.5, 4.8))


def plot_saved_and_deleted_articles():
    number_phase_2 = len(phase_2_df)
    number_deleted = len(deleted_df)
    x = [number_phase_2, number_deleted]
    label = ["Kept", "Deleted"]
    total_count = number_phase_2 + number_deleted
    colors = ['lightskyblue', "lightcoral"]
    plt.pie(x, startangle=90, colors=colors, frame=True, textprops=None,
            autopct=lambda p: '{:.0f}\n({:.0f}%)'.format((p / 100) * total_count, p), pctdistance=0.45)
    add_white_circle()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(label, loc="upper right", fancybox=True, shadow=True)
    plt.axis('equal')
    plt.title("Outcome of Manual Search Filtration")
    show_and_save("saved_vs_deleted.png")


def plot_model_type_distribution():
    df_grouped = phase_2_df.groupby(by=["Type of method/model"])
    hybrids = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    methods = {"Hybrid": 0}
    horizon = {"Hybrid": {'1': 0, '2': 0, '3': 0, 'Mix': 0}, "Multi-Agent": None, "Fundamental": None,
               "Reduced Form": None, "Statistical": None, "Data-driven": None}
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
    df = df.reindex(["Hybrid", "Data-driven", "Statistical", "Reduced Form", "Fundamental", "Multi-Agent"])
    df = df.fillna(0)
    print(df)
    df["2"] = df["2"] + df["Mix"]  # remove mix, add them to mid-term
    df = df[["1", "2", "3"]]
    print("Method per horizon no mix")
    print(df)
    print("\n")
    colors_ = ["mediumaquamarine", 'lightskyblue', "lightcoral"]
    df.plot(kind="barh", color=colors_, stacked=True, title="Classification of Articles", figsize=(8, 5), width=0.75)
    plt.legend(["Short Term", "Med. Term", "Long Term"], fancybox=True, shadow=True)
    plt.xlabel("Article Amount")
    plt.ylabel("Model Family")
    plt.tick_params(bottom=True, left=False, labelleft=True, labelbottom=True)
    values = [int(a) for a in df.sum(axis=1).tolist()]
    plt.xlim(0, max(values)*1.08)
    print(values)
    for i, v in enumerate(values):
        plt.text(v+12, i, str(v), color="black", size=9, ha='center')
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
    plt.bar(names, values, color=five_model_colors)
    plt.title("Hybrid Model Distribution")
    plt.ylabel("Article Amount")
    plt.xlabel("Model Family")
    for i, v in enumerate(values):
        plt.text(i, v + 1, str(v), color="black", size=8, ha='center')
    plt.ylim(0, max(values) * 1.1)
    plt.tick_params(bottom=False)
    show_and_save("hybrid_distribution.png")


def plot_probabilistic_pie():
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
    plt.pie(values, startangle=90, frame=True, textprops=None, colors=colors_,
            autopct=lambda p: '{:.0f}'.format((p / 100) * sum(values), p) if p > 0 else "", pctdistance=0.53)
    add_white_circle()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(names, loc="upper right", fancybox=True, shadow=True)
    plt.axis('equal')
    plt.title("Probabilistic Prediction per Model")
    show_and_save("probabilistic_output.png")


def plot_probabilistic_per_horizon():
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
    prob_proportion_df.plot(kind="bar", stacked=True, color=colors, title="Prediction Type per Horizon")
    plt.legend(["Point", "Prob."], fancybox=True, shadow=True, ncol=2)
    plt.xticks(range(len(replace_dict_horizon)), [v for v in replace_dict_horizon.values()], rotation=0)
    plt.tick_params(bottom=False)
    for i, v in enumerate([v[1] for v in prob_count.values()]):
        percent = int(prob_proportion[str(i + 1)][1] * 100)
        bar_string = "{}% ({})".format(percent, v)
        plt.text(i, 1.02, bar_string, color="black", size=8, ha='center')
    plt.ylabel("Proportion")
    plt.xlabel("Horizon")
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


def plot_data_driven_per_year():
    desired_model_type = ["5", "1, 5", "2, 5", "3, 5", "4, 5"]
    data_driven_df = phase_2_df[phase_2_df["Type of method/model"].isin(desired_model_type)]
    data_driven_year_df = data_driven_df.groupby(by=["Year"]).size().reset_index(name="Count")
    years = data_driven_year_df["Year"].tolist()
    article_count = data_driven_year_df["Count"].tolist()
    plt.bar(years, article_count, color="mediumaquamarine", label="Article amount")
    plt.title("Data-driven Articles per Year")
    plt.ylabel("Article Amount")
    plt.xlabel("Time")
    article_count_average = [(article_count[0] + article_count[1]) / 2]
    for i in range(1, len(years) - 1):
        prev = article_count[i - 1]
        this = article_count[i]
        next_ = article_count[i + 1]
        avg = (prev + this + next_) / 3
        article_count_average.append(avg)
    article_count_average.append((article_count[-1] + article_count[-2]) / 2)
    plt.plot(years, article_count_average, color="seagreen", linewidth=4, label="3 year mean")
    plt.legend(loc="upper left", fancybox=True, shadow=True)
    show_and_save("data_driven_articles_time.png")


def plot_model_type_within_data_driven():
    print("\nPlotting model type within data-driven articles..")
    df = pd.read_csv("data/data_driven_methods.csv")
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
    replace_keys = {"1": "FFNN", "2": "RNN", "3": "Fuzzy", "4": "SVM", "5": "Bio.", "6": "Ensamble", "7": "Other"}
    red = "lightcoral"
    blue = "lightskyblue"
    green = "mediumaquamarine"
    plt.bar(replace_keys.values(), count.values(), color=green)
    plt.ylim(0, 1.1 * max(count.values()))
    for i, v in enumerate(count.values()):
        plt.text(i, v + 5, str(v), color="black", size=8, ha='center')
    plt.ylabel("Article Amount")
    plt.xlabel("Model Type")
    plt.title("Data-driven Model Distribution")
    show_and_save("data_driven_method_dist.png")
    combi_count = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}
    combi_names = ["NN", "Simulation", "SVM", "Bio.", "Other"]
    combi_df = df_grouped.get_group("6")
    for index, row in combi_df.iterrows():
        model_list = row["Model"].split(", ")
        for m in model_list:
            combi_count[m] += 1
    print(combi_count)
    x = combi_count.values()
    five_greens = ["seagreen", "mediumaquamarine", "gainsboro", "limegreen", "palegreen"]
    plt.pie(x, colors=five_greens, autopct=lambda p: '{:.0f}'.format(p), pctdistance=0.53, frame=True)
    add_white_circle()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(combi_names, loc="upper right", fancybox=True, shadow=True)
    plt.title("Data-driven Ensamble Model Distribution")
    plt.axis('equal')
    show_and_save("ensamble_dist.png")


def plot_articles_per_year():
    phase_2_year_df = phase_2_df.groupby(by=["Year"]).size().reset_index(name="Count")
    years = phase_2_year_df["Year"].tolist()
    article_count = phase_2_year_df["Count"].tolist()
    set_plot_size()
    plt.bar(years, article_count, color="lightskyblue", label="Article amount")
    plt.title("Research Units per Year")
    plt.ylabel("Article Amount")
    plt.xlabel("Time")
    article_count_average = [(article_count[0] + article_count[1]) / 2]
    for i in range(1, len(years) - 1):
        prev = article_count[i - 1]
        this = article_count[i]
        next_ = article_count[i + 1]
        avg = (prev + this + next_) / 3
        article_count_average.append(avg)
    article_count_average.append((article_count[-1] + article_count[-2]) / 2)
    plt.plot(years, article_count_average, color="royalblue", linewidth=4, label="3 year mean")
    plt.legend(loc="upper left", fancybox=True, shadow=True)
    show_and_save("articles_per_year.png")

if __name__ == '__main__':
    print("Running methods..\n")
    # plot_saved_and_deleted_articles()
    # plot_model_type_distribution()
    # plot_hybrid_distribution()
    # plot_probabilistic_pie()
    # plot_probabilistic_per_horizon()
    plot_data_driven_per_year()
    # plot_model_type_within_data_driven()
    # plot_articles_per_year()
