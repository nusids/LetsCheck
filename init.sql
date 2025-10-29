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
    
CREATE TABLE tweets_incr (
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
    
CREATE TABLE users_incr (
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

CREATE UNIQUE INDEX "idx_tweet_id_incr" ON "tweets_incr" (
	"tweet_id"	ASC
);

CREATE UNIQUE INDEX "idx_user" ON "users" (
	"user_id"	ASC
);

CREATE UNIQUE INDEX "idx_user_incr" ON "users" (
	"user_id"	ASC
);
