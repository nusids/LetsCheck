// Unused after migrating to sqlite
const Utils = require('./Utils')
const SQL_MAX_VARIABLES = 500;
require('dotenv').config();
const {Pool} = require('pg')
const pool = new Pool({
    user: 'postgres',
    host: process.env.PG_HOST ? process.env.PG_HOST : '192.168.231.1',
    database: 'twatch',
    password: 'postgres',
    port: 5432,
})

const wait = (timeout) => {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve()
        }, timeout);
    });
}

class SearchUtils {
    constructor() {
        this.pending_requests = [];
        this.pending_results = new Map();
    }

    getPendingRequests() {
        return this.pending_requests;
    }

    removePendingRequest(id) {
        this.pending_requests = this.pending_requests.filter(r => r.id.toString() !== id)
        return true;
    }

    saveResult(id, re) {
        console.log(`Received query result for request ${id}`)
        this.pending_results.set(id, re);
    }

    getResult(id) {
        const re = this.pending_results.get(id);
        this.pending_results.delete(id);
        return re;
    }

    async fetchRequest(request) {
        request.id = Math.floor(Math.random() * 10 ** 12).toString();
        this.pending_requests.push(request);
        let waited = 0;
        while (waited < 60) {
            const reqId = request.id;
            if (this.pending_results.get(reqId)) {
                return this.getResult(reqId);
            } else {
                await wait(1000);
                // console.log(`Waiting for results for ${waited} seconds`);
                waited += 1;
            }
        }
        return {};
    }

    async fetchOverallTrend() {
        return this.fetchRequest({type: 'trend'});
    }

    async fetchTweetsById(ids, full = true) {
        return this.fetchRequest({type: 'tweets_by_id', ids: ids, full: full});
    }

    async fetchUsersById(ids) {
        return this.fetchRequest({type: 'users_by_id', ids: ids});
    }

    async fetchReplyTweetsByRootTweetId(ids) {
        return this.fetchRequest({type: 'reply_tweets_by_root_id', ids: ids});
    }

    async fetchTweetsMatchingText(queryRaw) {
        return this.fetchRequest({type: 'tweets_matching_text', queryRaw: queryRaw});
    }

    async fetchCustomRequest(req_params) {
        return this.fetchRequest(req_params);
    }

    async searchTweets(query) {

        try {
            console.log(`Received query "${query}"`);
            let retrievedTweetsIdSet = new Set();
            let retrievedTweets = await this.fetchTweetsMatchingText(query);
            console.log(`Retrieved ${retrievedTweets.length} rows from DB.`);
            if (retrievedTweets.length === 0) {
                return {isValid: false};
            }

            let replyOriginTweetsIds = new Set();
            let originalTweets = [];

            retrievedTweets.forEach(retrievedTweet => {
                retrievedTweetsIdSet.add(retrievedTweet.tweet_id);
            });

            retrievedTweets.forEach(retrievedTweet => {
                if (retrievedTweet.reply_parent_id === '-1') {
                    // This is an original tweet
                    originalTweets.push(retrievedTweet);
                } else if (retrievedTweet.reply_parent_id !== '-1') {
                    // This is a reply
                    if (!retrievedTweetsIdSet.has(retrievedTweet.reply_parent_id)) {
                        retrievedTweetsIdSet.add(retrievedTweet.reply_parent_id);
                        replyOriginTweetsIds.add(retrievedTweet.reply_parent_id)
                    }
                }
            });

            // Recursively fetch all reply origin tweets
            let replyOriginTweetItems = [];
            while (replyOriginTweetsIds.size > 0) {
                let [re] = await Promise.all([this.fetchTweetsById(Array.from(replyOriginTweetsIds), false)]);
                replyOriginTweetsIds.clear();
                re.forEach(replyTweetItem => {
                    // Add next level of reply parents
                    retrievedTweetsIdSet.add(replyTweetItem.tweet_id);
                    if (replyTweetItem.reply_parent_id !== "-1" && !retrievedTweetsIdSet.has(replyTweetItem.reply_parent_id)) {
                        replyOriginTweetsIds.add(replyTweetItem.reply_parent_id);
                    }
                });
                replyOriginTweetItems.concat(re);
            }

            // Add all retweet origins and preserve ascending unix order
            // let [retweetOriginTweetItems] = await Promise.all([fetchTweetsById(Array.from(retweetOriginTweetsIds), full = false)]);
            // retrievedTweets = retrievedTweets.concat(replyOriginTweetItems).concat(retweetOriginTweetItems);
            // replyOriginTweetItems.forEach(item => {
            //     const idx = Utils.binarySearch(retrievedTweets, item.unix, 'unix');
            // })

            console.log(`Total ${retrievedTweetsIdSet.size} tweets (including reply and retweet origins).`);

            // get data segmented by time
            console.log('Processing trend...');
            const trend = Utils.getTweetTrends(retrievedTweets);
            let progressiveCount = [];
            let currentSum = 0;
            // Cumulative count
            for (let i = 0; i < trend.length; i++) {
                currentSum += trend[i].y;
                progressiveCount.push({time: trend[i].x, count: currentSum});
            }

            // get most influential users and tweets by time
            let startTime = new Date();
            console.log('Processing segmented data...');

            const userMap = new Map();
            let thisSegmentTweetArray = [];
            let progressiveinfluentialUsers = [];
            let progressiveinfluentialTweets = [];
            let prevSegmentUsers = [];
            let prevSegmentTweets = [];
            let segmentId = 0;

            let selectedTweetsId = new Set();
            let selectedUsersId = new Set();

            for (let j = 0; j < retrievedTweets.length; j++) {
                if (segmentId >= progressiveCount.length) {
                    break;
                }

                // Advance to next time segment
                if (j > progressiveCount[segmentId].count) {
                    progressiveinfluentialUsers.push({
                        time: progressiveCount[segmentId].time,
                        users: prevSegmentUsers,
                    });
                    progressiveinfluentialTweets.push({
                        time: progressiveCount[segmentId].time,
                        tweets: prevSegmentTweets,
                    });
                    thisSegmentTweetArray.length = 0;
                    segmentId++;
                    j--;
                }

                // If this is the last tweet of the time segment
                if (j === progressiveCount[segmentId].count) {
                    let userArray = [];
                    userMap.forEach((value, key) => {
                        userArray.push({
                            screen_name: key,
                            id_str: value.id_str,
                            count: value.count,
                            followers_count: value.followers_count
                        });
                    });
                    userArray = userArray.sort((a, b) => b.count * b.followers_count - a.count * a.followers_count).slice(0, 10);
                    userArray.forEach(user => {
                        selectedUsersId.add(user.id_str);
                    })

                    progressiveinfluentialUsers.push({
                        time: progressiveCount[segmentId].time,
                        users: userArray,
                    });

                    let influentialTweets = thisSegmentTweetArray
                        .sort((a, b) => b.influence_score - a.influence_score).slice(0, 10);
                    influentialTweets.forEach(t => {
                        selectedTweetsId.add(t.tweet_id)
                    })

                    progressiveinfluentialTweets.push({
                        time: progressiveCount[segmentId].time,
                        tweets: influentialTweets.map(a => ({...a})),
                    });

                    prevSegmentUsers = userArray;
                    prevSegmentTweets = influentialTweets;

                    thisSegmentTweetArray.length = 0;
                    segmentId++;
                    j--;
                } else {
                    // Accumulate tweet details
                    let item = retrievedTweets[j];
                    if (userMap.has(item.user_name)) {
                        const temp = userMap.get(item.user_name);
                        temp.count += 1;
                        userMap.set(item.user_name, temp);
                    } else {
                        userMap.set(item.user_name, {
                            id_str: item.user_id,
                            count: 1,
                            followers_count: parseInt(item.user_followers_count)
                        });
                    }
                    thisSegmentTweetArray.push(item);
                }
            }

            for (; segmentId < progressiveCount.length; segmentId++) {
                progressiveinfluentialUsers.push({
                    time: progressiveCount[segmentId].time,
                    users: prevSegmentUsers,
                });
                progressiveinfluentialTweets.push({
                    time: progressiveCount[segmentId].time,
                    tweets: prevSegmentTweets,
                });
            }

            let now = new Date();
            console.log(`Segmented data processed in ${(now - startTime) / 1000} s`);

            // Get details
            const [selectedTweetsDetail, selectedUsersDetail] = await Promise.all(
                [this.fetchTweetsById(Array.from(selectedTweetsId)),
                    this.fetchUsersById(Array.from(selectedUsersId))]
            );

            selectedTweetsDetail.forEach(selectedTweet => {
                selectedTweet.user = {
                    'id_str': selectedTweet.user_id,
                    'name': selectedTweet.user_name,
                    'screen_name': selectedTweet.user_screen_name,
                    'location': selectedTweet.user_location,
                    'followers_count': selectedTweet.user_followers_count,
                    'profile_image_url': selectedTweet.user_profile_image_url,
                    'friends_count': selectedTweet.user_friends_count,
                    'verified': selectedTweet.user_is_verified === 'True',
                }
            });

            let selectedTweetsMap = new Map();
            selectedTweetsDetail.forEach(t => {
                selectedTweetsMap.set(t.tweet_id, t);
            })
            let selectedUsersMap = new Map();
            selectedUsersDetail.forEach(u => {
                selectedUsersMap.set(u.user_id, u);
            })

            let allTimeInfluentialTweets = selectedTweetsDetail
                .sort((a, b) => b.num_retweets + b.num_favourites - a.num_retweets - a.num_favourites).slice(0, 10);

            let allTimeInfluentialUsers = progressiveinfluentialUsers[progressiveinfluentialUsers.length - 1].users

            // compress data
            let progressiveData = [];
            for (let i = 0; i < progressiveCount.length; i++) {
                const influentialUsers = progressiveinfluentialUsers[i] ?
                    progressiveinfluentialUsers[i].users.map(u => ({...selectedUsersMap.get(u.id_str), count: u.count}))
                    : progressiveinfluentialUsers[i - 1].users.map(u => ({
                        ...selectedUsersMap.get(u.id_str),
                        count: u.count
                    }))
                const influentialTweets = progressiveinfluentialTweets[i] ? progressiveinfluentialTweets[i].tweets.map(t => selectedTweetsMap.get(t.tweet_id))
                    : progressiveinfluentialTweets[i - 1].tweets.map(t => selectedTweetsDetail.get(t.tweet_id))
                progressiveData.push({
                    time: progressiveCount[i].time,
                    count: progressiveCount[i].count,
                    influentialUsers: influentialUsers,
                    influentialTweets: influentialTweets,
                });
            }

            // Only show top 100 threads on diffusion network
            const originalTweetIds = originalTweets.sort((a, b) => b.influence_score - a.influence_score)
                .slice(0, 100).map(x => x.tweet_id);

            console.log("Finished retrieving data.")
            console.log("-------------------------")
            return {
                isValid: true,
                event: {
                    query,
                    isFromSearch: true,
                    influentialUsers: allTimeInfluentialUsers,
                    influentialTweets: allTimeInfluentialTweets,
                    trend: trend,
                    desc: query,
                },
                diffusionIds: originalTweetIds,
                progData: progressiveData,
                tweetInfo: selectedTweetsDetail,
                userInfo: selectedUsersDetail,
            }
        } catch (e) {
            return {isValid: false};
        }
    }
}


module.exports = {
    SearchUtils
}