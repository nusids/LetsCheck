import React, { Component } from 'react';
import PropTypes from 'prop-types';
// import * as numeral from 'numeral';
// import _ from 'underscore';
import { connect } from 'react-redux';
// import {
//   FlexibleWidthXYPlot, XAxis, YAxis, LineSeries, Hint,
// } from 'react-vis';
// import moment from 'moment';
import Loader from 'react-loader-spinner';
// import { getTweetTrends } from '../utils/util';
import { fetchTweetsByIds } from '../actions/tweetAction';
import { fetchUsersByIds } from '../actions/userAction';
import TweetCard from './TweetCard';
import * as styles from './DiffusionNetworkDetails.scss';

const ReplyType = Object.freeze({
  agreed: 'agreed',
  disagreed: 'disagreed',
  appealForMoreInfo: 'appeal-for-more-information',
  comment: 'comment',
});

const DisplyMode = Object.freeze({
  none: 0,
  clusterInfo: 1,
  mouseOverReplyTweet: 2,
  mouseOverRetweet: 3,
  mouseOverCluster: 4,
  selectedRetweet: 5,
  selectedReplyTweet: 6,
  outsideCluster: 7,
  mouseOverSourceTweet: 8,
});

const getDescriptionFromAnnotation = (annotation) => {
  const type = annotation.responsetype_vs_source;

  switch (type) {
    case ReplyType.agreed:
      return 'agrees with';
    case ReplyType.disagreed:
      return 'disagrees with';
    case ReplyType.appealForMoreInfo:
      return 'appeals more information from';
    default:
      return 'is a comment of';
  }
};

class DiffusionNetworkDetails extends Component {
  constructor(props) {
    super(props);

    this.mode = DisplyMode.none;

    this.updateMode();
  }

  componentDidMount() {
    this.updateMode();
    this.updateTweet();
  }

  componentDidUpdate() {
    this.updateMode();
    this.updateTweet();
    this.updateUser();
  }

  // onImageError(e) {
  //   e.target.src = defaultProfileImg;
  // }

  onAccountClicked(screenName) {
    window.open(`https://twitter.com/${screenName}`, '_blank');
  }

  updateMode() {
    const {
      isInCluster, mouseoverTweet, sourceTweet, selectedTweet,
    } = this.props;

    if (!isInCluster && !mouseoverTweet) {
      this.mode = DisplyMode.outsideCluster;
    } else if (!isInCluster && mouseoverTweet) {
      this.mode = DisplyMode.mouseOverCluster;
    } else if (isInCluster) {
      if (mouseoverTweet && sourceTweet && mouseoverTweet.id_str === sourceTweet.id_str) {
        this.mode = DisplyMode.mouseOverSourceTweet;
      } else if ((!mouseoverTweet && !selectedTweet && sourceTweet)
        || (selectedTweet && sourceTweet && selectedTweet.id_str === sourceTweet.id_str)) {
        this.mode = DisplyMode.clusterInfo;
      } else if (selectedTweet && selectedTweet.annotation.isReply) {
        this.mode = DisplyMode.selectedReplyTweet;
      } else if (selectedTweet && selectedTweet.annotation.isRetweet) {
        this.mode = DisplyMode.selectedRetweet;
      } else if (mouseoverTweet && mouseoverTweet.annotation.isReply) {
        this.mode = DisplyMode.mouseOverReplyTweet;
      } else if (mouseoverTweet && mouseoverTweet.annotation.isRetweet) {
        this.mode = DisplyMode.mouseOverRetweet;
      }
    }
  }

  updateTweet() {
    const {
      getTweets, tweetInfo, sourceTweet, selectedTweet, mouseoverTweet,
      // eventName,
    } = this.props;
    let tweet = null;

    if (this.mode === DisplyMode.mouseOverCluster || this.mode === DisplyMode.mouseOverReplyTweet) {
      tweet = mouseoverTweet.data;
    } if (this.mode === DisplyMode.clusterInfo) {
      tweet = sourceTweet.data;
    } else if (this.mode === DisplyMode.selectedReplyTweet) {
      tweet = selectedTweet.data;
    }

    if (!tweet) {
      return;
    }

    if (!tweetInfo[tweet.id_str]) {
      getTweets([tweet.id_str]);
    }
  }

  updateUser() {
    const {
      // getUsers, userInfo,
      influentialAccountIds, tweetInfo, mouseoverTweet, selectedTweet, sourceTweet,
    } = this.props;
    let ids = [];

    if (this.mode === DisplyMode.clusterInfo) {
      ids = [...influentialAccountIds.map(x => x.id_str)];

      if (tweetInfo[sourceTweet.id_str] && tweetInfo[sourceTweet.id_str].user) {
        ids.push(tweetInfo[sourceTweet.id_str].user.id_str);
      }
    } else if (this.mode === DisplyMode.selectedRetweet || this.mode === DisplyMode.mouseOverRetweet) {
      // ids = retweetsInCluster.map(item => item.id_str);
    } else if (this.mode === DisplyMode.mouseOverCluster || this.mode === DisplyMode.mouseOverReplyTweet) {
      if (tweetInfo[mouseoverTweet.id_str] && tweetInfo[mouseoverTweet.id_str].user) {
        ids = [tweetInfo[mouseoverTweet.id_str].user.id_str];
      }
    } else if (this.mode === DisplyMode.selectedReplyTweet) {
      if (tweetInfo[selectedTweet.id_str] && tweetInfo[selectedTweet.id_str].user) {
        ids = [tweetInfo[selectedTweet.id_str].user.id_str];
      }
    }

    // ids = _.uniq(ids);
    // const idsToFetch = ids.filter(x => !userInfo[x]);

    // if (idsToFetch.length > 0) {
    //   getUsers(idsToFetch);
    // }
  }

  render() {
    const {
      mouseoverTweet,
      sourceTweet,
      // tweetInfo,
      selectedTweet,
      // influentialAccountIds,
      numNodes,
      numTweets,
      allTweets,
      isInCluster,
    } = this.props;

    if (numNodes === 0) {
      return (<div>
        <div className={styles.loader}>
          <Loader type="RevolvingDot" color="#000000" height="50" />
        </div>
        <span className={styles.message}>Building diffusion network...</span>
              </div>);
    }

    this.updateMode();

    let desc = '';
    // let trendElement = null;

    if (isInCluster && allTweets.length >= 2) {
      // const trend = getTweetTrends(allTweets.map(item => item.unix));
      // let nearestPoint = null;
      // let timeToHighlight = 0;
      // let retweetTrend = null;

      // if (trend.length >= 2) {
      //   if (this.mode === DisplyMode.mouseOverReplyTweet
      //   || this.mode === DisplyMode.selectedReplyTweet
      //   || this.mode === DisplyMode.mouseOverSourceTweet) {
      //     const step = trend[1].x - trend[0].x;
      //     const tweetId = this.mode === DisplyMode.selectedReplyTweet ? selectedTweet.id_str : mouseoverTweet.id_str;
      //     timeToHighlight = allTweets.find(item => item.tweet_id === tweetId).unix;
      //     nearestPoint = trend.find(item => Math.abs(item.x - timeToHighlight) < step);
      //   }
      //
      //   if (this.mode === DisplyMode.selectedRetweet || this.mode === DisplyMode.mouseOverRetweet) {
      //     retweetTrend = getTweetTrends(allTweets.filter(x => x.isRetweet).map(x => x.unix));
      //   }
      //
      //   trendElement = (
      //     <div className={styles.trend}>
      //       <div className={styles.title}>{`Thread timeline - ${numTweets} tweets`}</div>
      //       <FlexibleWidthXYPlot
      //         height={150}
      //         animation
      //       >
      //         <XAxis
      //           hideLine
      //           tickSize={0}
      //           tickTotal={6}
      //           tickFormat={v => moment.unix(v).format('ha MMM DD')}
      //           style={{
      //             text: { fontSize: '0.8rem' },
      //           }}
      //         />
      //         <YAxis
      //           hideLine
      //           tickTotal={4}
      //           tickSize={0}
      //           style={{
      //             text: { fontSize: '0.8rem' },
      //           }}
      //         />
      //         <LineSeries
      //           curve="curveCatmullRom"
      //           color="#6A93B1"
      //           data={trend}
      //         />
      //         {retweetTrend ? (
      //           <LineSeries
      //             curve="curveCatmullRom"
      //             color="#F29B1D"
      //             data={retweetTrend}
      //           />
      //         ) : null}
      //         {nearestPoint ? (
      //           <Hint
      //             value={{ x: nearestPoint.x, y: 0 }}
      //             align={{ vertical: 'top', horizontal: 'right' }}
      //           >
      //             <div className={styles.hintContainer}>
      //               <div className={styles.textContainer}>{moment.unix(timeToHighlight).format('HH:mm MMM DD, YYYY')}</div>
      //               <div className={styles.indicator} />
      //             </div>
      //           </Hint>
      //         ) : null}
      //         {nearestPoint ? (
      //           <Hint
      //             value={nearestPoint}
      //             align={{ vertical: 'top', horizontal: 'right' }}
      //           >
      //             <div className={styles.dot} />
      //           </Hint>
      //         ) : null}
      //       </FlexibleWidthXYPlot>
      //     </div>
      //   );
      // }
    }

    if (this.mode === DisplyMode.selectedRetweet) {
      desc = `There are ${selectedTweet.annotation.count} retweets from source tweet.`;
    } else if (this.mode === DisplyMode.mouseOverRetweet) {
      desc = `There are ${mouseoverTweet.annotation.count} retweets from source tweet.`;
    } else if (this.mode === DisplyMode.mouseOverReplyTweet) {
      desc = `This tweet ${getDescriptionFromAnnotation(mouseoverTweet.annotation)} source tweet`;
    } else if (this.mode === DisplyMode.selectedReplyTweet) {
      desc = `This tweet ${getDescriptionFromAnnotation(selectedTweet.annotation)} source tweet`;
    } else if (this.mode === DisplyMode.mouseOverCluster) {
      desc = `This is a source tweet of the thread: ${mouseoverTweet.num_replies_shown} ${mouseoverTweet.num_replies_shown > 1 ? 'replies' : 'reply'}, ${mouseoverTweet.num_retweets_shown} ${mouseoverTweet.num_retweets_shown > 1 ? 'retweets' : 'retweet'}.`;
    }

    const descComponent = desc ? <div className={styles.tweetDesc}>{desc}</div> : null;

    if (this.mode === DisplyMode.outsideCluster) {
      return (
        <div className={styles.container}>
          <p>Showing top 100 threads.</p>
          <div className={styles.infoTitle}>
            <span className={styles.number}>{numNodes}</span>
            <span className={styles.text}>Threads, </span>
            <span className={styles.number}>{numTweets}</span>
            <span className={styles.text}>Tweets</span>
          </div>
        </div>
      );
    }

    if (this.mode === DisplyMode.selectedRetweet) {
      return (
        <div className={styles.container}>
          {/* {trendElement} */}
          {descComponent}
        </div>
      );
    }

    if (this.mode === DisplyMode.selectedReplyTweet) {
      const { data } = selectedTweet;
      const userData = {
        id_str: data.user_id,
        name: data.user_name,
        screen_name: data.user_screen_name,
        location: data.user_location,
        followers_count: data.user_followers_count,
        friends_count: data.user_friends_count,
        is_verified: data.user_is_verified,
        profile_image_url: data.user_profile_image_url,
      };
      return (
        <div className={styles.container}>
          {/* {trendElement} */}
          {descComponent}
          <div className={styles.singleTweetContainer}>
            <TweetCard data={data} user={userData} withShadow />
          </div>
        </div>
      );
    }

    if (this.mode === DisplyMode.mouseOverCluster) {
      const { data } = mouseoverTweet;
      const userData = {
        id_str: data.user_id,
        name: data.user_name,
        screen_name: data.user_screen_name,
        location: data.user_location,
        followers_count: data.user_followers_count,
        friends_count: data.user_friends_count,
        is_verified: data.user_is_verified,
        profile_image_url: data.user_profile_image_url,
      };
      return (
        <div className={styles.container}>
          {descComponent}
          <div className={styles.subtitle}>Click on the thread to view more.</div>
          <div className={styles.singleTweetContainer}>
            <TweetCard
              data={data}
              user={userData}
              withShadow
            />
          </div>
        </div>
      );
    }

    if (this.mode === DisplyMode.clusterInfo
        || this.mode === DisplyMode.mouseOverReplyTweet
        || this.mode === DisplyMode.mouseOverRetweet
        || this.mode === DisplyMode.mouseOverSourceTweet) {
      const tweetData = sourceTweet.data;
      const userData = {
        id_str: tweetData.user_id,
        name: tweetData.user_name,
        screen_name: tweetData.user_screen_name,
        location: tweetData.user_location,
        followers_count: tweetData.user_followers_count,
        friends_count: tweetData.user_friends_count,
        is_verified: tweetData.user_is_verified,
        profile_image_url: tweetData.user_profile_image_url,
      };
      return (
        <div className={styles.container}>
          {/* {trendElement} */}
          <div className={styles.tweetAccountContainer}>
            <div className={styles.singleTweetContainer}>
              <div className={styles.title}>Source Tweet</div>
              <TweetCard
                data={tweetData}
                user={userData}
                withShadow
              />
            </div>
            {/* <div className={styles.accountsContainer}> */}
            {/*  <div className={styles.title}>Influential accounts</div> */}
            {/*  <div className={styles.content}> */}
            {/*    {influentialAccountIds.map(x => getAccountItem(userInfo[x.id_str], x.followers_count))} */}
            {/*  </div> */}
            {/* </div> */}
          </div>
          {/* <div className={styles.urlContainer}> */}
          {/*  <div className={styles.title}>Relevant links</div> */}
          {/*  <div className={styles.urlContent}> */}
          {/*    {sourceTweet.annotation.links.map((item, i) => <LinkPreviewCard url={item.link} key={i} />)} */}
          {/*  </div> */}
          {/* </div> */}
        </div>
      );
    }

    return (
      <div>Error here</div>
    );
  }
}

DiffusionNetworkDetails.propTypes = {
  // eventName: PropTypes.string.isRequired,
  tweetInfo: PropTypes.object.isRequired,
  // userInfo: PropTypes.object.isRequired,
  getTweets: PropTypes.func.isRequired,
  // getUsers: PropTypes.func.isRequired,
  isInCluster: PropTypes.bool.isRequired,
  numNodes: PropTypes.number.isRequired,
  numTweets: PropTypes.number.isRequired,
  allTweets: PropTypes.array.isRequired,
  influentialAccountIds: PropTypes.array,
  sourceTweet: PropTypes.object,
  mouseoverTweet: PropTypes.object,
  selectedTweet: PropTypes.object,
};

DiffusionNetworkDetails.defaultProps = {
  sourceTweet: null,
  mouseoverTweet: null,
  selectedTweet: null,
  influentialAccountIds: [],
};

export default connect(
  state => ({
    tweetInfo: state.tweetInfo,
    userInfo: state.userInfo,
  }),
  dispatch => ({
    getTweets: ids => dispatch(fetchTweetsByIds(ids)),
    getUsers: ids => dispatch(fetchUsersByIds(ids)),
  }),
)(DiffusionNetworkDetails);
