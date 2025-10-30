CREATE TABLE tweets (
        unix integer,
        num_retweets integer,
        num_favourites integer,
        tweet_id text NOT NULL,
        user_id text,
        reply_parent_id text,
        text text,
        location text,
        root_tweet_id text
    );
    
    
CREATE TABLE users (
        user_followers_count integer,
        user_friends_count integer,
        user_id text NOT NULL,
        user_name text,
        user_screen_name text,
        user_location text,
        user_is_verified text,
        user_profile_image_url text
    );
    

CREATE UNIQUE INDEX "idx_tweet_id" ON "tweets" (
	"tweet_id"	ASC
);

CREATE INDEX "idx_root_tweet_id" ON "tweets" (
	"root_tweet_id"	ASC
);

CREATE UNIQUE INDEX "idx_user" ON "users" (
	"user_id"	ASC
);

CREATE VIRTUAL TABLE tweets_fts USING fts4(content="tweets", tweet_id, text);

CREATE TRIGGER t_bu BEFORE UPDATE ON tweets BEGIN
  DELETE FROM tweets_fts WHERE docid=old.rowid;
END;
CREATE TRIGGER t_bd BEFORE DELETE ON tweets BEGIN
  DELETE FROM tweets_fts WHERE docid=old.rowid;
END;

CREATE TRIGGER t_au AFTER UPDATE ON tweets BEGIN
  INSERT INTO tweets_fts(docid, tweet_id, text) VALUES(new.rowid, new.tweet_id, new.text);
END;
CREATE TRIGGER t_ai AFTER INSERT ON tweets BEGIN
  INSERT INTO tweets_fts(docid, tweet_id, text) VALUES(new.rowid, new.tweet_id, new.text);
END;
