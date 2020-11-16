# script for getting the difference between two searches
import pandas as pd


def get_left_not_in_right_articles(first_df, sec_df):
    first_titles = first_df["Title"].tolist()
    sec_titles = sec_df["Title"].tolist()
    diff_titles = [t for t in first_titles if t not in sec_titles]
    diff_df = first_df[first_df["Title"].isin(diff_titles)]
    return diff_df


def get_output_articles():
    original_df = pd.read_csv("first_1012.csv")
    slr_phase_1_df = pd.read_csv("slr_phase_1.csv", sep=";", encoding="utf-8")
    deleted_phase_1_orginal_df = get_left_not_in_right_articles(original_df, slr_phase_1_df)
    correct_df = pd.read_csv("full_1288.csv")
    print("Length of correct search: {}".format(len(correct_df)))
    in_correct_not_in_original = get_left_not_in_right_articles(correct_df, original_df)
    print("Number of new articles not in original search: {}".format(len(in_correct_not_in_original)))
    correct_removing_deleted_phase_1_df = get_left_not_in_right_articles(correct_df, deleted_phase_1_orginal_df)
    print("Length after removing irrelevant articles: {}".format(len(correct_removing_deleted_phase_1_df)))
    deleted_articles_phase_1 = get_left_not_in_right_articles(correct_df, correct_removing_deleted_phase_1_df)
    deleted_articles_phase_1.to_csv("deleted_phase_1.csv", index=False)
    print("Number of deleted articles: {}".format(len(deleted_articles_phase_1)))
    new_articles_df = get_left_not_in_right_articles(correct_removing_deleted_phase_1_df, slr_phase_1_df)
    print("Number of new articles not in 650: {}".format(len(new_articles_df)))
    in_650_not_in_correct = get_left_not_in_right_articles(slr_phase_1_df, correct_removing_deleted_phase_1_df)
    print("Number of articles in 650 that is not in correct search: {}".format(len(in_650_not_in_correct)))
    updated_640 = get_left_not_in_right_articles(slr_phase_1_df, in_650_not_in_correct)
    print("Length of 650 after removing false articles: {}".format(len(updated_640)))
    updated_932_df = updated_640.append(in_correct_not_in_original, ignore_index=True)
    del updated_932_df['Link']
    updated_932_df.to_csv("slr_phase_2.csv", index=False)
    print("Length after adding new articles: {}".format(len(updated_932_df)))



if __name__ == '__main__':
    get_output_articles()
