import pandas as pd
import sys
import re
import glob
import os 

files = glob.glob("*.csv") # get all the files ending with .csv in the current directory

count = 0
for file in files:
    try:
        if (count % 100 == 0):
            print(count)
        count += 1
        df = pd.read_csv(file, error_bad_lines=False) # skip lines with faulty data
        
        # create columns for the DataFrame
        df.columns = list('tweet_id_db,tweet_time_db,retweet_user_id_db,source_user_id_db,retweet_user_name_db,source_user_name_db,retweet_user_follower_count_db,retweet_user_location_db,source_user_location_db,'.split(','))
        
        def replace_non_numeric_characters(sequence): # eliminate non numeric characters in tweet IDs, user IDs etc, and convert to string
            return re.sub('[^\d]', '', str(sequence))

        df['tweet_id_db_str'] = df['tweet_id_db'].apply(replace_non_numeric_characters)
        df['retweet_user_id_db_str'] = df['retweet_user_id_db'].apply(replace_non_numeric_characters)
        df['source_user_id_db_str'] = df['source_user_id_db'].apply(replace_non_numeric_characters)
        
        df = df.drop(columns=['tweet_id_db', 'retweet_user_id_db', 'source_user_id_db']) # drop columns
        OUTPUT_FILE_NAME = str(count) + '_retweet.out'

        df.to_csv(
            OUTPUT_FILE_NAME,
            index = False, 
            header = False)
    except:
        print(file + ' got error, skipped')
        continue
    

