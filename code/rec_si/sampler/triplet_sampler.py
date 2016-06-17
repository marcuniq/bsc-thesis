import numpy as np
import pandas as pd


class TripletSampler(object):
    def __init__(self, unique_movie_ids):
        self.unique_movie_ids = unique_movie_ids

    def sample(self, train, factor):
        grouped = train.groupby('user_id')

        def create_triplet_for_user(df):
            non_rated = np.setdiff1d(self.unique_movie_ids, df['movie_id'].unique())
            sampled_non_rated = np.random.choice(non_rated, size=factor*len(df))
            new_df = pd.DataFrame(df)
            new_df = new_df.append([df]*(factor-1), ignore_index=True)
            new_df['movie_id2'] = sampled_non_rated
            return new_df

        df_tuples = grouped.apply(create_triplet_for_user).reset_index(drop=True)
        return df_tuples
