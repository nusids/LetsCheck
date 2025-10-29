CREATE TABLE IF NOT EXISTS cord (
    idx integer PRIMARY KEY,
    cord_uid text,
    snippet text,
    encoded blob
);

CREATE TABLE IF NOT EXISTS cord_metadata (
    cord_uid text PRIMARY KEY,
    title text,
    authors text,
    journal text,
    url text
);

CREATE TABLE IF NOT EXISTS news (
	idx integer PRIMARY KEY,
	title text,
	url text,
	date text,
	snippet text,
	encoded blob
);

CREATE INDEX IF NOT EXISTS "cord_uid_idx" ON "cord" (
	"cord_uid"	ASC
);

CREATE INDEX IF NOT EXISTS "news_url_idx" ON "news" (
	"url"	ASC
);
