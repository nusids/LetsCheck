#!/bin/sh

while true; do
	mkdir /home/liuhang/logs
	cd /home/liuhang/logs
	find * -type f -mtime +7 -delete
	cd /home/liuhang/letscheck-twitter-crawler/
	/home/liuhang/letscheck-twitter-crawler/venv/bin/python /home/liuhang/letscheck-twitter-crawler/tsv_to_db.py > /home/liuhang/logs/tweet-crawler-$(date -I) 2>&1
	sleep 86400
done
