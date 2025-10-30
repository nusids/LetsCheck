import pandas as pd
import sys
import re
import glob
import os 

files = glob.glob("*.csv") # get all the files ending with .csv in the current directory

count = 0
for file in files:
    try:
        if (count % 100 == 0): # for debug purposes
            print(count)
        count += 1
        df = pd.read_csv(file)

        # create columns for the DataFrame
        df.columns = list('tweet_id_db,tweet_time_db,is_retweet_db,tweet_content_db,user_name_db,user_id_db,user_follower_count_db,user_location_db,'.split(','))

        def replace_non_numeric_characters(sequence): # eliminate non numeric characters in tweet IDs, user IDs etc, and convert to string
            return str(re.sub('[^\d]', '', str(sequence)))

        df['tweet_id_db_str'] = df['tweet_id_db'].apply(replace_non_numeric_characters)
        df['user_id_db_str'] = df['user_id_db'].apply(replace_non_numeric_characters)

        def clean_tweet_content(content):
            def replace(string_in, src, dest):
                return dest.join(string_in.split(src)) # replace all occurences of src by dest
            result = content
            replacements = {
                '\\xe2\\x80\\x99': "'",
                '\\xc3\\xa9': 'e',
                '\\xe2\\x80\\x90': '-',
                '\\xe2\\x80\\x91': '-',
                '\\xe2\\x80\\x92': '-',
                '\\xe2\\x80\\x93': '-',
                '\\xe2\\x80\\x94': '-',
                '\\xe2\\x80\\x95': '?',
                '\\xe2\\x80\\x96': '-',
                '\\xe2\\x80\\x97': '-',
                '\\xe2\\x80\\x98': "'",
                '\\xe2\\x80\\x99': "'",
                '\\xe2\\x80\\x9a':"'",
                '\\xe2\\x80\\x9b':"'",
                '\\xe2\\x80\\x9c':'"',
                '\\xe2\\x80\\x9d':'"',
                '\\xe2\\x80\\x9e':'"',
                '\\xe2\\x80\\x9f':'"',
                '\\xe2\\x80\\xa4':'.',
                '\\xe2\\x80\\xa5':'..',
                '\\xe2\\x80\\xa6':'...',
                '\\xe2\\x80\\xa7':'?',
                '\\xe2\\x80\\xb2':"'",
                '\\xe2\\x80\\xb3':"'",
                '\\xe2\\x80\\xb4':"'",
                '\\xe2\\x80\\xb5':"'",
                '\\xe2\\x80\\xb6':"'",
                '\\xe2\\x80\\xb7':"'",
                '\\xe2\\x81\\xba':"+",
                '\\xe2\\x81\\xbb':"-",
                '\\xe2\\x81\\xbc':"=",
                '\\xe2\\x81\\xbd':"(",
                '\\xe2\\x80\\x8b':"(",
                '\\n':" ",
                "\"":'"',
                '\\xe2\\x81\\xbd':"(",
                '\\xe2\\x81\\xbe':")",
                '\xe2\x80\x99': "'",
                '\xc3\xa9': 'e',
                '\xe2\x80\x90': '-',
                '\xe2\x80\x91': '-',
                '\xe2\x80\x92': '-',
                '\xe2\x80\x93': '-',
                '\xe2\x80\x94': '-',
                '\xe2\x80\x95': '?',
                '\xe2\x80\x96': '-',
                '\xe2\x80\x97': '-',
                '\xe2\x80\x98': "'",
                '\xe2\x80\x99': "'",
                '\xe2\x80\x9a':"'",
                '\xe2\x80\x9b':"'",
                '\xe2\x80\x9c':'"',
                '\xe2\x80\x9d':'"',
                '\xe2\x80\x9e':'"',
                '\xe2\x80\x9f':'"',
                '\xe2\x80\xa4':'.',
                '\xe2\x80\xa5':'..',
                '\xe2\x80\xa7':'?',
                '\xe2\x80\xa6':'...',
                '\xe2\x80\xb2':"'",
                '\xe2\x80\xb3':"'",
                '\xe2\x80\xb4':"'",
                '\xe2\x80\xb5':"'",
                '\xe2\x80\xb6':"'",
                '\xe2\x80\xb7':"'",
                '\xe2\x81\xba':"+",
                '\xe2\x81\xbb':"-",
                '\xe2\x81\xbc':"=",
                '\xe2\x81\xbd':"(",
                '\xe2\x80\x8b':"(",
                '\xe2\x81\xbd':"(",
                '\xe2\x81\xbe':")",
            }

            for key in replacements:
                result = replace(result, key, replacements[key]) # clean non-numeric characters

            result = result[2: -1] # remove the first two characters, b', and the last '
            return result

        df['tweet_content_db_cleaned'] = df['tweet_content_db'].apply(clean_tweet_content)
        df['tweet_content_db_cleaned_lower_case'] = df['tweet_content_db_cleaned'].apply(lambda string: string.lower())
        df = df.drop(columns=['tweet_content_db', 'tweet_id_db', 'user_id_db']) # drop columns

        OUTPUT_FILE_NAME = str(count) + '.out'

        df.to_csv(
            OUTPUT_FILE_NAME,
            index = False, 
            header = False)
    except:
        print(file + 'got error, skipped')
        continue