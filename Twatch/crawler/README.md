# Twatch tweet crawler
This folder contains scripts that downloads the "dehydrated" version of COVID-19 related tweets as found in [The Panacea Lab dataset](https://github.com/thepanacealab/covid19_twitter), hydrate them, and store the relevant information into an SQLITE database.

## Current status
- The crawler script is located on A100. The service is managed by `/etc/init.d/tweet-crawler`. To check crawling status, go to `$HOME/logs`.
- The service should always be running; the script itself will sleep for 24 hours after downloading the latest data.
- When the script has finished downloading a new batch of data, an empty file named `new_tweets_available` will be generated. EXALIT searches for this file every day at 2am using a cron job. If new updates are found, EXALIT will copy the Twatch DB from A100 and restart Twatch backend.
- In the event that the `tweet-crawler` service is down, the script `dgx-a100-status-check.sh` on EXALIT should detect it within 20 minutes and restart it. Refer to `misc/dgx-a100-status-check.sh`.

## What each file does
- `crawl.sh`: sets up & clean up the log folder, calls the Python crawler script, and sleeps for 24 hours before checking for new update again. This script takes care of the entire crawling process and is intended to be used in a systemd service unit file, or other similar service managers.
- `init.sql`: the database schema of the output SQLITE database. It should be used in the event of creating a new DB from scratch.
- `last_processed_date`: keeps track of what was the latest set of daily tweets downloaded from Panacea Lab repository. Don't delete.
- `processed_chunks`: keeps track of which chunk of a single dataset file has been processed. This was meant for big tsv files containing all historical tweets, which were >1GB in size, but is currently still in use, so do not delete.
- `requirements.txt`: install required Python packages: `pip install -r requirements.txt`
- `tsv_to_db.py`: the main script that does the downloading, processing and storing of processed tweets.
- `tsv_to_postgres_[errors_]log`: stores output from `tsv_to_db.py`. The output database used to be Postgres hence the filename.
- `twitter_auth_info.py`: put Twitter API keys here
- `whitelist`: tweets from accounts whose handles are found here will not be left out even though they do not get enough likes/retweets. Currently empty but still used in script, so do not delete.