import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import db_password
import time

#data location
file_dir = "C:/Users/jason/Documents/Data_analytics/raw_data/"

# open/read all data
with open(f'{file_dir}wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)

kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv')
ratings = pd.read_csv(f'{file_dir}ratings.csv')


def etl_autobot(wiki_movies_raw, kaggle_metadata, ratings):
    """Perform Extract, Transform Load for three data sets: wiki movie scrape, kaggle metadata, and kaggle ratings data."""
    
    
    # ensure each movie has a director and imbd link. remove TV series ('No. of episodes')
    wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                  and 'No. of episodes' not in movie]
    
    
    #Large function to clean the wiki data set
    def clean_movie(movie):

        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)

        #create a non-destructive copy    
        movie = dict(movie)


        #loop through atl title keys, removing and placing in one key alt_titles
        alt_titles = {}
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune–Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        #add alt_titles dict to movie being cleaned
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles


        # merge column name for same columns

        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')

        return movie   
    
    #clean our data by row and create a data frame
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)

    
    # use regular expression to get the imbd_id from imbd link and store it in a new column
    # Drop duplicates by imbd_id
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    
    # Drop columns with mostly null data
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns 
                            if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
    
    
    # Parse box office data
    # get box office set for cleaning and converting, joins lists together
    box_office = wiki_movies_df['Box office'].dropna() 
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
    
    #replaces hyphen with $ to fix dollar range issues, re forms:
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    # build re for “$123.4 million” (or billion)
    form_one = r"\$\s*\d+\.?\d*\s*[mb]illi?on"
    #second form, “$123,456,789.”
    form_two = r"\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)"
    
    #function to parse the forms from strings to floats 
    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
            # remove dollar sign and " million". convert to float and multiply by a million
            s = re.sub(r'\$|\s|[a-zA-Z]','', s)
            value = float(s) * 10**6
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " billion".convert to float and multiply by a billion
            s = re.sub(r'\$|\s|[a-zA-Z]','', s)
            value = float(s) * 10**9
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas, convert to float
            s = re.sub(r'\$|,','', s)
            value = float(s)
            return value

        # otherwise, return NaN
        else:
            return np.nan

    #apply parse dollars fn to all strings. Drop outdated column
    try:
        wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
        wiki_movies_df.drop('Box office', axis=1, inplace=True)
    except:
        wiki_movies_df.rename(columns={"Box office": "box_office"})
        print(" error parsing box office data")
        
    # Parse budget data
    budget = wiki_movies_df['Budget'].dropna()
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    #remove any values between a dollar sign and a hyphen (for budgets given in ranges), and citation references
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    
    #Parse budget data through parse dollars fn, drop outdated column
    try:
        wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
        wiki_movies_df.drop('Budget', axis=1, inplace=True)
    except:
        wiki_movies_df.rename(columns={"Budget": "budget"})
        print(" error parsing budget data")
    
    
    # Parse release date
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    
    # Full month name, one- to two-digit day, four-digit year (i.e., January 1, 2000)
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    # Four-digit year, two-digit month, two-digit day, with any separator (i.e., 2000-01-01)
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    # Full month name, four-digit year (i.e., January 2000)
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    # Four-digit year
    date_form_four = r'\d{4}'
    
    #Parse release date and use build in infer datetime format to convert, drop outdated column
    try:
        wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
        wiki_movies_df.drop('Release date', axis=1, inplace=True)
    except:
        wiki_movies_df.rename(columns={"Release date": "release_date"})
        print(" error parsing release date")
        
    
    # Parse Runnint time
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    #capture forms hours min, and mins
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    # convert to numberic values. errors='coerce' handles empty strings 
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    #convert everythin to mins, append to df, drop outdated column
    try:
        wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
        wiki_movies_df.drop('Running time', axis=1, inplace=True)
    except:
        wiki_movies_df.rename(columns={"Running time": "running_time"})
        print(" error parsing running time")        
    
    #WIKI DATA TRANSORM COMPLETE
    
    # Clean Kaggle data kaggle_metadata and ratings

    # Drop adult films and column
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
    
    #convert to correct dtypes
    try:
        kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
    except:
        print("error converting kaggle video dtype, some data must not be of type bool") 
    try:
        kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    except:
        print("error converting kaggle budget dtype, some data must have characters")
        
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    try:
        kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
    except:
        print("error converting kaggle release date to datetime, must be a bad data point in the dataset.")
    try:
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    except:
        print("error converting ratings timestamp to datetime dtype, bad data point(s) in ratings dataset")
    
    #merge data frames
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
    # Competing data:
    # Wiki                     Movielens                Resolution
    #--------------------------------------------------------------------------
    # title_wiki               title_kaggle             drop wikipedia
    # running_time             runtime                  keep kaggle, fill 0 w wiki
    # budget_wiki              budget_kaggle            keep kaggle, fill 0 w wiki
    # box_office               revenue                  keep kaggle, fill 0 w wiki
    # release_date_wiki        release_date_kaggle      drop wiki
    # Language                 original_language        drop wiki
    # Production company(s)    production_companies     drop wiki
    
    #drop bad data
    movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)
    
    #fill and drop
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column], axis=1)
        df.drop(columns=wiki_column, inplace=True)
    
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # Reorganize df 
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]
    # Rename df columns
    movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)
    
    
    # Organize the ratings dataset
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) \
                .pivot(index='movieId',columns='rating', values='count')
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
    
    # Merge it all together
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)
    
    # LOAD!
    # Connect to SQL
    db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
    engine = create_engine(db_string)
    # transfer data frames, replacing data if it exists
    movies_df.to_sql(name='movies', con=engine, if_exists='replace')
    
    rows_imported = 0
    start_time = time.time()
    
    for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='replace')
        rows_imported += len(data)
        print(f'Done. {time.time() - start_time} total seconds elapsed')
    
    
    return movies_with_ratings_df


etl_autobot(wiki_movies_raw, kaggle_metadata, ratings)

