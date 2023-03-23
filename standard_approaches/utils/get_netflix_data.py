import pandas as pd
import tarfile
from datetime import datetime


def data2timestamp(t):
    t = list(map(int, t.split('-')))
    return int(datetime(*t).timestamp())


def get_netflix_data(gz_file, get_ratings=True, sample_frac=0.05, random_state=42):
    movie_data = []
    movie_inds = []
    with tarfile.open(gz_file) as tar:
        if get_ratings:
            training_data = tar.getmember('download/training_set.tar')
            # maybe try with threads, e.g.
            # https://stackoverflow.com/questions/43727520/speed-up-json-to-dataframe-w-a-lot-of-data-manipulation
            with tarfile.open(fileobj=tar.extractfile(training_data)) as inner:
                for item in inner.getmembers():
                    if item.isfile():
                        f = inner.extractfile(item.name)
                        df = pd.read_csv(f)
                        movieid = df.columns[0]
                        movie_inds.append(int(movieid[:-1]))
                        movie_data.append(df[movieid])

    data = None
    if movie_data:
        data = pd.concat(movie_data, keys=movie_inds)
        data = data.reset_index().iloc[:, :4].rename(columns={'level_0': 'movieid',
                                                              'level_1': 'userid',
                                                              'level_2': 'rating',
                                                              0: 'timestamp'})
        data.loc[:, 'timestamp'] = data['timestamp'].apply(data2timestamp)

        data = data.sort_values(by='timestamp')
        n_samples = int(data.shape[0] * sample_frac)
        data = data.loc[:n_samples]

    return data
