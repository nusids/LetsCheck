import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Loader from 'react-loader-spinner';
import * as numeral from 'numeral';
import defaultProfileImg from '../assets/default-user.png';
import tweetsImg from '../assets/tweets.svg';
import followersImg from '../assets/followers.svg';
import * as styles from './AccountCard.scss';
import { toHttps } from '../utils/util';

class AccountCard extends Component {
  constructor(props) {
    super(props);

    this.onCardClicked = this.onCardClicked.bind(this);
    this.onImageError = this.onImageError.bind(this);
  }

  onCardClicked() {
    const { data } = this.props;

    if (data.isValidUser) {
      window.open(`https://twitter.com/${data.user_screen_name}`, '_blank');
    }
  }

  onImageError(e) {
    e.target.src = defaultProfileImg;
  }

  render() {
    const { tweetCount } = this.props;
    let { data } = this.props;

    if (!data || data.isLoading) {
      return (
        <div className={styles.loader}>
          <Loader type="RevolvingDot" color="#000000" height="30" />
        </div>
      );
    }

    if (!data.isValidUser) {
      data = {
        profile_image_url: defaultProfileImg,
        followers_count: '...',
        name: 'Account Suspended',
        screen_name: '...',
      };
    }

    const bottom = (
      <div className={styles.bottom}>
        <div className={styles.section}>
          <img className={styles.icon} src={followersImg} alt="" />
          <div className={styles.right}>
            <div className={styles.count}>{numeral(data.user_followers_count).format('0a')}</div>
            <div className={styles.desc}>followers</div>
          </div>
        </div>
        <div className={styles.section}>
          <img className={styles.icon} src={tweetsImg} alt="" />
          <div className={styles.right}>
            <div className={styles.count}>{tweetCount}</div>
            <div className={styles.desc}>tweets</div>
          </div>
        </div>
      </div>
    );

    return (
      <div
        className={styles.card}
        onClick={this.onCardClicked.bind(this)}
      >
        <div className={styles.top}>
          <img className={styles.icon} src={toHttps(data.user_profile_image_url)} alt={data.user_name} onError={this.onImageError} />
          <div className={styles.name}>{data.user_name}</div>
          <div className={styles.screenName}>{`@${data.user_screen_name}`}</div>
        </div>
        {bottom}
      </div>
    );
  }
}

AccountCard.propTypes = {
  tweetCount: PropTypes.number.isRequired,
  data: PropTypes.object,
};

AccountCard.defaultProps = {
  data: null,
};

export default AccountCard;
