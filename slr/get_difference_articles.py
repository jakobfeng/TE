# script for getting the difference between two searches
import pandas as pd


def get_left_not_in_right_articles(first_df, sec_df):
    first_titles = first_df["Title"].tolist()
    sec_titles = sec_df["Title"].tolist()
    diff_titles = [t for t in first_titles if t not in sec_titles]
    diff_df = first_df[first_df["Title"].isin(diff_titles)]
    return diff_df


def get_output_articles():
    original_df = pd.read_csv("data\\old\\first_1012.csv")
    slr_phase_1_df = pd.read_csv("data\\old\\slr_phase_1.csv", sep=";", encoding="utf-8")
    deleted_phase_1_orginal_df = get_left_not_in_right_articles(original_df, slr_phase_1_df)
    correct_df = pd.read_csv("data\\full_1288.csv")
    print("Length of correct search: {}".format(len(correct_df)))
    in_correct_not_in_original = get_left_not_in_right_articles(correct_df, original_df)
    print("Number of new articles not in original search: {}".format(len(in_correct_not_in_original)))
    correct_removing_deleted_phase_1_df = get_left_not_in_right_articles(correct_df, deleted_phase_1_orginal_df)
    print("Length after removing irrelevant articles: {}".format(len(correct_removing_deleted_phase_1_df)))
    deleted_articles_phase_1 = get_left_not_in_right_articles(correct_df, correct_removing_deleted_phase_1_df)
    #deleted_articles_phase_1.to_csv("deleted_phase_1_604.csv", index=False)
    print("Number of deleted articles: {}".format(len(deleted_articles_phase_1)))
    new_articles_df = get_left_not_in_right_articles(correct_removing_deleted_phase_1_df, slr_phase_1_df)
    print("Number of new articles not in 650: {}".format(len(new_articles_df)))
    in_650_not_in_correct = get_left_not_in_right_articles(slr_phase_1_df, correct_removing_deleted_phase_1_df)
    print("Number of articles in 650 that is not in correct search: {}".format(len(in_650_not_in_correct)))
    updated_640 = get_left_not_in_right_articles(slr_phase_1_df, in_650_not_in_correct)
    print("Length of 650 after removing false articles: {}".format(len(updated_640)))
    updated_932_df = updated_640.append(in_correct_not_in_original, ignore_index=True)
    del updated_932_df['Link']
    #updated_932_df.to_csv("slr_phase_2.csv", index=False)
    print("Length after adding new articles: {}".format(len(updated_932_df)))
    # ----------------------------------------------------------------
    phase_1_682_df = pd.read_csv("data/phase_2_682.csv")
    all_data_driven_df = pd.read_csv("data/old/data_driven_all.csv")
    horizons = ["1", "2", "1, 2", "1, 3", "2, 3", "1, 2, 3"]
    all_short_medium_term = phase_1_682_df[phase_1_682_df["Horizon"].isin(horizons)]
    print("Length of short and medium: {}".format(len(all_short_medium_term)))
    all_data_driven_2 = pd.read_csv("data/old/data_driven_all.csv")
    data_driven_long = get_left_not_in_right_articles(all_data_driven_2, all_short_medium_term)
    print("Length of data driven long term: {}".format(len(data_driven_long)))
    data_driven_short_medium = get_left_not_in_right_articles(all_data_driven_2, data_driven_long)
    print("Length of data driven short and medium: {}".format(len(data_driven_short_medium)))
    # data_driven_short_medium.to_csv("data\\data_driven_short_medium.csv", index=False, sep=",")
    wrong_markets = ["2", "3", "2, 3"]
    data_driven_wrong_markets = phase_1_682_df[phase_1_682_df["Market"].isin(wrong_markets)]
    data_driven_short_medium = pd.read_csv("data/old/data_driven_short_medium.csv")
    data_driven_short_medium_spot = get_left_not_in_right_articles(data_driven_short_medium, data_driven_wrong_markets)
    print("Lenght og data driven correct hor. correct market: {}".format(len(data_driven_short_medium_spot)))
    # ata_driven_short_medium_spot.to_csv("data\\data_driven_short_medium_spot.csv", index=False)
    # ---------------------------
    des_horizon = ["1, 2", "2", "2, 3"]
    medium_term_df = phase_1_682_df[phase_1_682_df["Horizon"].isin(des_horizon)]
    des_output = ["2", "1, 2"]
    medium_term_prob_df = medium_term_df[medium_term_df["Output"].isin(des_output)]
    medium_term_prob_df.to_csv("data\\medium_term_prob.csv", index=False)




if __name__ == '__main__':
    get_output_articles()
