/* eslint-disable prefer-destructuring */
import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Loader from 'react-loader-spinner';
import { TwitterTweetEmbed } from 'react-twitter-embed';

import defaultProfileImg from '../assets/default-user.png';
import * as styles from './TweetCard.scss';

class TweetCard extends Component {
  constructor(props) {
    super(props);

    this.onTweetClicked = this.onTweetClicked.bind(this);
    this.onProfileClicked = this.onProfileClicked.bind(this);
    this.onImageError = this.onImageError.bind(this);
  }

  onTweetClicked() {
    const { data, user } = this.props;

    if (data.user) {
      window.open(
        `https://twitter.com/${data.user.screen_name}/status/${data.tweet_id}`,
        '_blank',
      );
    } else if (user) {
      window.open(
        `https://twitter.com/${user.screen_name}/status/${data.tweet_id}`,
        '_blank',
      );
    }
  }

  onProfileClicked(e) {
    e.stopPropagation();

    const { data, user } = this.props;
    if (user) {
      window.open(`https://twitter.com/${user.screen_name}`, '_blank');
    } else if (data.user) {
      window.open(`https://twitter.com/${data.user.screen_name}`, '_blank');
    }
  }

  onImageError(e) {
    e.target.src = defaultProfileImg;
  }

  render() {
    /* Because of Twitter API terms and conditions, custom rendering of tweets are not allowed. My previous Twitter
       account was suspended because of this.
       Using embedded tweets instead.
     */
    const {
      data, onLoad,
    } = this.props;
    let { user } = this.props;
    if (!data) {
      return [];
    }

    if (!user) {
      user = data.user;
    }

    const loader = (
      <div className={styles.loader}>
        <Loader type="RevolvingDot" color="#000000" height="30" />
      </div>
    );

    if (!data || data.isLoading || (user && user.isLoading)) {
      return loader;
    }

    if (!user) {
      user = {
        profile_image_url: defaultProfileImg,
        followers_count: 0,
        name: 'Account suspended',
        screen_name: 'undefined',
      };
    }

    return (
      <div className={styles.tweet}>
        <TwitterTweetEmbed
          tweetId={data.tweet_id}
          placeholder={loader}
          onLoad={onLoad}
          options={{
            conversation: 'none',
            cards: 'hidden',
            width: 275,
            maxWidth: 275,
            height: 450,
            maxHeight: 450,
          }}
        />
      </div>
    );
  }
}

TweetCard.propTypes = {
  onLoad: PropTypes.func,
  data: PropTypes.object,
  user: PropTypes.object,
};

TweetCard.defaultProps = {
  data: null,
  user: null,
  onLoad: () => {},
};

export default TweetCard;
