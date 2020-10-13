import pandas as pd

path = "..\\data\\slr\\"


def get_slr_df_name(name):
    full_path = str(path + name + ".csv")
    df = pd.read_csv(full_path, sep=",", header=0)
    return df


def get_ratio_between(first, second):
    first_df = get_slr_df_name(str(first))
    second_df = get_slr_df_name(str(second))
    count = 0
    hits = 0
    index = first_df.columns.get_loc("DOI")
    print("Index: " + str(index))
    for row in first_df.iterrows():
        count += 1
        print(count)
        doi = row[1][index]
        for r in second_df.iterrows():
            d = r[1][index]
            if doi == d:
                print("hit")
                hits +=1
    print("\nNumber of articles: " + str(count))
    print("Number of hits: " + str(hits))

def view_duplicates(name):
    df = get_slr_df_name(str(name))
    count = 0
    hits = 0
    index = df.columns.get_loc("Title")
    for row in df.iterrows():
        count += 1
        print(count)
        title = row[1][index]
        for r in df.iterrows():
            if row[0] != r[0]:
                t = r[1][index]
                if t == title:
                    print("hit")
                    hits +=1
    print("\nNumber of articles: " + str(count))
    print("Number of hits: " + str(hits))

if __name__ == '__main__':
    #get_ratio_between(783, 943)
    view_duplicates(930)