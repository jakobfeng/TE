# script for getting the difference between two searches
import pandas as pd


def get_left_not_in_right_articles(first_df, sec_df):
    first_titles = first_df["Title"].tolist()
    sec_titles = sec_df["Title"].tolist()
    diff_titles = [t for t in first_titles if t not in sec_titles]
    diff_df = first_df[first_df["Title"].isin(diff_titles)]
    return diff_df


def get_all_ci_methods_to_csv(phase_2_df):
    methods = ["5", "1, 5", "2, 5", "3, 5", "4, 5"]
    ci_df = phase_2_df[phase_2_df["Type of method/model"].isin(methods)]
    ci_df.to_csv("data/ci_methods_not_classified.csv", index=False, sep=",")


def extend_ci_methods_classified():
    ci_new = pd.read_csv("data\\ci_methods_not_classified.csv")
    ci_old = pd.read_csv("data/ci_methods.csv")
    new_articles = get_left_not_in_right_articles(ci_new, ci_old)
    in_prev_not_in_new = get_left_not_in_right_articles(ci_old, ci_new)
    ci_old_updated = get_left_not_in_right_articles(ci_old, in_prev_not_in_new)
    ci_old_updated = ci_old_updated.append(new_articles, ignore_index=True)
    ci_old_updated = ci_old_updated[["Title", "Abstract", "Author Keywords", "Index Keywords", "Approach", "Model"]]
    ci_old_updated.to_csv("data\\ci_methods.csv", index=False)

def get_data_driven_short_medium_spot_to_csv(phase_2_df):
    horizons = ["1", "2", "1, 2", "1, 3", "2, 3", "1, 2, 3"]
    all_short_medium_term = phase_2_df[phase_2_df["Horizon"].isin(horizons)]
    print("Length of short and medium: {}".format(len(all_short_medium_term)))
    all_data_driven = pd.read_csv("data/ci_methods.csv")
    data_driven_long = get_left_not_in_right_articles(all_data_driven, all_short_medium_term)
    print("Length of data driven long term: {}".format(len(data_driven_long)))
    data_driven_short_medium = get_left_not_in_right_articles(all_data_driven, data_driven_long)
    print("Length of data driven short and medium: {}".format(len(data_driven_short_medium)))
    wrong_markets = ["2", "3", "2, 3"]
    data_driven_wrong_markets = phase_2_df[phase_2_df["Market"].isin(wrong_markets)]
    data_driven_short_medium = pd.read_csv("data/old/data_driven_short_medium.csv")
    data_driven_short_medium_spot = get_left_not_in_right_articles(data_driven_short_medium, data_driven_wrong_markets)
    print("Lenght og data driven correct hor. correct market: {}".format(len(data_driven_short_medium_spot)))
    data_driven_short_medium_spot.to_csv("data\\ci_short_medium_spot.csv", index=False)


def get_medium_term_prob_to_csv(phase_2_df):
    des_horizon = ["1, 2", "2", "2, 3"]
    medium_term_df = phase_2_df[phase_2_df["Horizon"].isin(des_horizon)]
    des_output = ["2", "1, 2"]
    medium_term_prob_df = medium_term_df[medium_term_df["Output"].isin(des_output)]
    medium_term_prob_df.to_csv("data\\medium_term_prob.csv", index=False)


def add_document_type_to_phase_2(correct_df, phase_2_df):
    phase_2_df["Document Type"] = None
    for index, row in phase_2_df.iterrows():
        row_from_full = correct_df[correct_df["Title"]==row["Title"]].head(1)
        doc_type = str(row_from_full["Document Type"]).split()[1]
        if doc_type == "Review":
            doc_type = "Article"
        phase_2_df.loc[index, "Document Type"] = doc_type
    phase_2_df.to_csv("data\\phase_2_676.csv", index=False)


def compare_ci():
    from_phase_two = pd.read_csv("data\\ci_methods_not_classified.csv")
    ci_methods = pd.read_csv("data\\ci_methods_classified.csv")
    not_in_all = get_left_not_in_right_articles(ci_methods, from_phase_two)
    print("Articles not in phase 2 but in all ci: {}".format(len(not_in_all)))
    for index, row in not_in_all.iterrows():
        print(row["Title"])
    print("All: {}, phase 2 ci: {}".format(len(ci_methods), len(from_phase_two)))
    duplicates = get_title_of_duplicates(ci_methods)
    print(duplicates)


def merge_ci_df():
    ci_methods = pd.read_csv("data\\ci_methods_classified.csv")
    ci_approaches = ci_methods[["Title", "Approach", "Model"]]
    from_phase_two = pd.read_csv("data\\ci_methods_not_classified.csv")
    from_phase_two = from_phase_two.merge(ci_approaches, how="inner", on="Title")
    from_phase_two.to_csv("data\\ci_methods_classified.csv", index=False)


def get_title_of_duplicates(df):
    duplicates = []
    names = df["Title"].tolist()
    for n in names:
        count = 0
        for name in names:
            if n == name:
                count += 1
        if count > 1:
            duplicates.append(n)
    return duplicates


def write_statistical_to_file(phase_2):
    desired_model = ["4", "1, 4", "2, 4", "3, 4", "4, 5"]
    stat = phase_2[phase_2["Type of method/model"].isin(desired_model)]
    print("Length of stat phase 2: {}".format(len(stat)))
    stat.to_csv("data\\stat_methods.csv", index=False)



def get_output_articles():
    correct_df = pd.read_csv("data\\full_1288.csv")
    phase_2_df = pd.read_csv("data\\phase_2_670.csv")
    deleted_phase_1_df = pd.read_csv("data\\deleted_phase_1_618.csv")
    print("Search {}, phase 2 {}, deleted {}\n".format(len(correct_df), len(phase_2_df), len(deleted_phase_1_df)))
    merge_ci_df()


if __name__ == '__main__':
    get_output_articles()
