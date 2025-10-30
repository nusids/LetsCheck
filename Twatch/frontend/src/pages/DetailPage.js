import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { withRouter } from 'react-router';
import queryString from 'query-string';
import Loader from 'react-loader-spinner';
import Popover from 'react-tiny-popover';
import StackGrid from 'react-stack-grid';
import VisibilitySensor from 'react-visibility-sensor';

import { fetchSearchData } from '../actions/searchAction';
import {
  fetchEventByName, fetchProgressiveDataByName, fetchDiffusionData,
} from '../actions/detailsAction';
import { fetchUsersByIds } from '../actions/userAction';
import { fetchTweetsByIds } from '../actions/tweetAction';
import TweetCard from '../components/TweetCard';
import AccountCard from '../components/AccountCard';
import InteractiveTrend from '../components/InteractiveTrend';
import DiffusionNetwork from '../components/DiffusionNetwork';
import infoImg from '../assets/info.svg';
import styles from './DetailPage.scss';
import 'antd/lib/input-number/style/css';
import { getMax, getMin } from '../utils/util';

const CLICK_MARGIN = 100000;

const INFO_ITEMS = Object.freeze({
  influentialTweets: {
    index: 0,
    title: 'Influential Tweets',
  },
  influentialUsers: {
    index: 1,
    title: 'Influential Accounts',
  },
  diffusionNetwork: {
    index: 4,
    title: 'Diffusion Network',
  },
});

class DetailPage extends Component {
  constructor(props) {
    super(props);

    this.state = {
      selectedInfo: 0,
      isPlaying: false,
      currentTime: -1,
      currentMinTime: -1,
      currentMaxTime: -1,
      isTimeSelected: false,
      isMapInfoOpen: false,
      deadline: {
        day: null,
        hours: null,
      },
    };

    const { location: { search } } = this.props;
    const query = queryString.parse(search);

    this.eventName = query.event;
    this.searchQuery = query.q;
    this.onInfoOptionUpdate = this.onInfoOptionUpdate.bind(this);
    this.setAccountGridRef = this.setAccountGridRef.bind(this);
    this.setTweetGridRef = this.setTweetGridRef.bind(this);
    this.updateGridLayout = this.updateGridLayout.bind(this);
    this.onCurrentTimeChanged = this.onCurrentTimeChanged.bind(this);
    this.onCurrentTimeRangeChanged = this.onCurrentTimeRangeChanged.bind(this);
    this.onMapInfoClicked = this.onMapInfoClicked.bind(this);
    this.onPlayButtonClicked = this.onPlayButtonClicked.bind(this);
    this.onTimeCloseButtonClicked = this.onTimeCloseButtonClicked.bind(this);
    this.onCurrentTimeSelected = this.onCurrentTimeSelected.bind(this);
    this.onFactCardClicked = this.onFactCardClicked.bind(this);
    this.onDeadlineSet = this.onDeadlineSet.bind(this);
    this.onDeadLineUpdated = this.onDeadLineUpdated.bind(this);
    this.onMaxDeadLineCalculated = this.onMaxDeadLineCalculated.bind(this);
    this.onRedirectToTIBClicked = this.onRedirectToTIBClicked.bind(this);

    this.tweetGrid = null;
    this.accountGrid = null;
    this.progDataPreprocessed = false;
    this.threadInfoPreprocessed = false;
    this.xyDataProcessed = false;
    // this.itemDataLoaded = false;
    this.selectedSourceTweetIdToTIB = null;
    this.progDataMap = new Map();
    this.xyMap = new Map();
    this.userFollowersMap = new Map();
    this.tweetUserMap = new Map();
    this.overallThreadInfo = [];
    this.threadMap = new Map();
    this.sortedData = [];
  }

  componentDidMount() {
    const {
      getEvent,
      getProgData,
      isFromSearch,
      getSearchData,
    } = this.props;

    if (isFromSearch) {
      getSearchData(this.searchQuery);
      return;
    }

    getEvent(this.eventName);
    getProgData(this.eventName);
  }

  // eslint-disable-next-line no-unused-vars
  componentDidUpdate(prevProps, prevState) {
    const {
      progData, currentEvent,
    } = this.props;

    if (progData.length > 0 && !this.progDataPreprocessed) {
      this.preprocessProgData(progData);
    }

    if (!currentEvent) {
      return;
    }

    if (!this.xyDataProcessed) {
      this.preprocessXYData();
    }

    this.updateGridLayout();
  }

  onFactCardClicked(link) {
    window.open(link, '_blank');
  }

  onInfoOptionUpdate(idx) {
    const { selectedInfo } = this.state;
    const { getDiffusionData, diffusionData } = this.props;

    if (selectedInfo === idx || idx === 2 || idx === 3) {
      return;
    }

    if ((idx === 4 || idx === 5) && (!diffusionData || diffusionData.length === 0)) {
      // Load diffusion network data
      this.setState({
        currentTime: -1,
        currentMinTime: -1,
        currentMaxTime: -1,
      });
      getDiffusionData(this.props.diffusionIds);
    }

    this.setState({
      selectedInfo: idx,
    });
  }

  onCurrentTimeChanged(time, isSelected) {
    this.setState({
      currentTime: time,
      isTimeSelected: isSelected || false,
    });
    const { currentMinTime, currentMaxTime } = this.state;
    if (time < currentMinTime - CLICK_MARGIN || time > currentMaxTime + CLICK_MARGIN) {
      this.setState({
        currentMinTime: -1,
        currentMaxTime: -1,
      });
    }
  }

  onCurrentTimeRangeChanged(min, max) {
    this.setState({
      currentMinTime: min,
      currentMaxTime: max,
    });
  }

  onCurrentTimeSelected() {
    this.setState({
      isTimeSelected: true,
    });
  }

  onDeadLineUpdated(day, hour) {
    this.setState({
      deadline: {
        day,
        hour,
      },
    });
  }

  onMapInfoClicked() {
    const { isMapInfoOpen } = this.state;

    this.setState({
      isMapInfoOpen: !isMapInfoOpen,
    });
  }

  onPlayButtonClicked() {
    const { isPlaying } = this.state;

    this.setState({
      isPlaying: !isPlaying,
    });
  }

  onTimeCloseButtonClicked() {
    this.setState({
      currentTime: -1,
      isTimeSelected: false,
    });
  }

  onDeadlineSet(v, isDay) {
    const { deadline } = this.state;
    let newDay = deadline.day;
    let newHour = deadline.hour;

    if (isDay) {
      newDay = v;
    } else {
      newHour = v;
    }

    this.setState({
      deadline: {
        day: newDay,
        hour: newHour,
      },
    });
  }

  onMaxDeadLineCalculated(day, hour) {
    const carry = (hour + 1) >= 24 ? 1 : 0;
    const newDay = day + carry;
    const newHour = hour + (carry > 0 ? 0 : 1);
    this.setState({
      deadline: {
        day: newDay,
        hour: newHour,
      },
    });
  }

  onRedirectToTIBClicked(sourceTweetId) {
    this.selectedSourceTweetIdToTIB = sourceTweetId;
  }

  setTweetGridRef(grid) {
    this.tweetGrid = grid;
  }

  setAccountGridRef(grid) {
    this.accountGrid = grid;
  }

  updateGridLayout() {
    if (this.accountGrid) {
      this.accountGrid.updateLayout();
    }

    if (this.tweetGrid) {
      this.tweetGrid.updateLayout();
    }

    window.setTimeout(this.updateGridLayout, 100);
  }

  selectInfluentialTweetsAndUsers() {
    const {
      currentTime,
      currentMinTime,
      currentMaxTime,
      isTimeSelected,
    } = this.state;

    const selectedInfluTweets = new Set();
    const selectedInfluUsers = new Set();

    if (this.progDataPreprocessed && isTimeSelected && currentMaxTime - currentMinTime > CLICK_MARGIN) {
      // Selecting from a range
      // This condition ensures that currentMaxTime and currentMinTime are far apart enough; otherwise treat it as
      // a single timestamp
      const availableTimes = Array.from(this.progDataMap.keys());
      let selectedTime = -1;
      for (let i = 0; i < availableTimes.length; i += 1) {
        selectedTime = availableTimes[i];
        if (selectedTime < currentMinTime) {
          continue;
        } else if (selectedTime >= currentMaxTime) {
          break;
        }

        if (this.progDataMap.has(selectedTime)) {
          this.progDataMap.get(selectedTime).influentialTweets.forEach(t => selectedInfluTweets.add(t));
          this.progDataMap.get(selectedTime).influentialUsers.forEach(u => selectedInfluUsers.add(u));
        }
      }
    } else if (this.progDataPreprocessed && isTimeSelected && currentTime > 0) {
      // Selecting at a point
      const availableTimes = Array.from(this.progDataMap.keys());
      let selectedTime = -1;
      let lastDiff = -1;
      for (let i = 0; i < availableTimes.length; i += 1) {
        selectedTime = availableTimes[i];
        lastDiff = Math.abs(currentTime - selectedTime);
        if (selectedTime >= currentTime) {
          if (lastDiff > 0 && Math.abs(currentTime - selectedTime) > lastDiff) {
            selectedTime = availableTimes[i - 1];
          }
          break;
        }
      }
      if (this.progDataMap.has(selectedTime)) {
        this.progDataMap.get(selectedTime).influentialTweets.forEach(t => selectedInfluTweets.add(t));
        this.progDataMap.get(selectedTime).influentialUsers.forEach(u => selectedInfluUsers.add(u));
      }
    } else {
      this.props.currentEvent.influentialTweets.forEach(t => selectedInfluTweets.add(t));
      this.props.currentEvent.influentialUsers.forEach(t => selectedInfluUsers.add(t));
    }

    return {
      tweets: Array.from(selectedInfluTweets).sort((a, b) => b.num_retweets + b.num_favourites - a.num_retweets - a.num_favourites),
      users: Array.from(selectedInfluUsers).sort((a, b) => b.num_retweets + b.num_favourites - a.num_retweets - a.num_favourites),
    };
  }

  preprocessProgData() {
    const { progData } = this.props;

    if (progData.length === 0) {
      return;
    }

    this.progDataPreprocessed = true;

    progData.forEach((item) => {
      this.progDataMap.set(item.time, {
        count: item.count,
        influentialUsers: item.influentialUsers,
        influentialTweets: item.influentialTweets,
      });
    });
  }

  preprocessXYData() {
    const { currentEvent } = this.props;

    currentEvent.trend.forEach((item) => {
      this.xyMap.set(item.x, item.y);
    });
    this.xyDataProcessed = true;
  }

  preprocessOverallThreadInfo() {
    const { diffusionData } = this.props;

    let min = Number.MAX_VALUE;
    let max = Number.MIN_VALUE;

    diffusionData.forEach((item) => {
      const tweets = [item];
      tweets.push(...item.children);
      tweets.push(...item.replies);

      tweets.forEach((t) => {
        this.userFollowersMap.set(t.user_id, t.user_followers_count);
        this.tweetUserMap.set(t.tweet_id, t.user_id);
      });

      const allTimes = item.children.concat(item.replies).concat([item]).map(x => x.unix);
      const minTime = getMin(allTimes);
      const maxTime = getMax(allTimes);
      min = Math.min(minTime, min);
      max = Math.max(maxTime, max);

      this.overallThreadInfo.push({
        source_id_str: item.tweet_id,
        minTime,
        maxTime,
        x0: minTime,
        x: maxTime,
      });
      this.threadMap.set(item.tweet_id, item);
    });

    this.overallThreadInfo.sort((a, b) => a.minTime - b.minTime);

    this.overallThreadInfo = this.overallThreadInfo.map((item, i) => ({
      ...item,
      y0: i,
      y: i + 0.9,
      index: i,
      color: '#6A93B1',
    }));

    this.minTime = min;
    this.maxTime = max;

    // get sorted data
    this.overallThreadInfo.forEach((item) => {
      this.sortedData.push({
        ...this.threadMap.get(item.source_id_str),
        minTime: item.minTime,
        maxTime: item.maxTime,
      });
    });
    this.threadInfoPreprocessed = true;
  }

  render() {
    const {
      currentEvent,
      isLoadingEvent,
      isLoadingProgData,
      diffusionData,
      userInfo,
    } = this.props;
    const {
      selectedInfo,
      currentTime,
      isMapInfoOpen,
      isTimeSelected,
    } = this.state;

    if (isLoadingEvent
      || isLoadingProgData) {
      return (
        <div className={styles.page}>
          <div className={styles.loader}>
            <Loader type="RevolvingDot" color="#000000" height="50" />
          </div>
          <span className={styles.message}>Loading data...</span>
        </div>
      );
    }

    if (!currentEvent) {
      return (
          <div className={styles.page}>
            <span className={styles.message}>
              No tweets found, please try another search term.
            </span>
          </div>
      );
    }

    const influ = this.selectInfluentialTweetsAndUsers();
    const influTweets = influ.tweets;
    const influUsers = influ.users;

    // info 0
    const trend = (
      <div className={styles.section} key={0}>
        <VisibilitySensor onChange={this.onTrendOutOfView}>
          <InteractiveTrend
            data={currentEvent.trend}
            currentTimeValue={{
              x: currentTime,
              y: this.xyMap.get(currentTime),
            }}
            isTimeSelected={isTimeSelected}
            onChange={this.onCurrentTimeChanged}
            onRangeChange={this.onCurrentTimeRangeChanged}
          />
        </VisibilitySensor>
      </div>
    );

    // info 1
    const tweetItems = influTweets.slice(0, 10).map(
      (item, i) => (
        <div className={styles.item} key={i}>
          <TweetCard
            data={item}
            onLoad={this.updateGridLayout}
          />
        </div>),
    );
    const tweetsInvolved = (
      <div className={styles.section} key={1}>
        <StackGrid
          columnWidth="24%"
          gutterWidth={24}
          gutterHeight={24}
          monitorImagesLoaded
          gridRef={grid => this.setTweetGridRef(grid)}
        >
          {tweetItems}
        </StackGrid>
      </div>
    );

    // info 2
    const accountItems = influUsers.slice(0, 10).map(item => (item.id_str ? userInfo[item.id_str] : userInfo[item.user_id])).map(
      (user, i) => (
        <div className={styles.item} key={i}>
          <AccountCard
            data={user}
            tweetCount={influUsers[i].count}
          />
        </div>),
    );
    const accountsInvolved = (
      <div className={styles.section} key={2}>
        <StackGrid
          columnWidth="20%"
          gutterWidth={20}
          gutterHeight={20}
          monitorImagesLoaded
          gridRef={grid => this.setAccountGridRef(grid)}
        >
          {accountItems}
        </StackGrid>
      </div>
    );

    // info 3
    const map = (
      <div className={styles.section} key={3}>
        <div className={styles.title}>
          <div className={styles.text}>Geographic propagation</div>
          <div className={styles.info}>
            <Popover
              isOpen={isMapInfoOpen}
              position="bottom"
              content={(
                <div className={styles.popover}>As location is only available if user explicitly shares his/her location while sending the tweet, and only a few users choose to do so, so by default this shows the location user displays in their Twitter profile instead.</div>
              )}
            >
              <img onClick={this.onMapInfoClicked} alt="info" src={infoImg} />
            </Popover>
          </div>
        </div>
      </div>
    );

    // info 4
    const diffusionNetwork = (
      <div className={styles.section} key={4}>
        <DiffusionNetwork
          data={diffusionData}
          currentTime={currentTime}
          eventName={this.eventName}
          onRedirectToTIBClicked={this.onRedirectToTIBClicked}
        />
      </div>
    );

    const infoOptions = Object.values(INFO_ITEMS)
      .sort((a, b) => a.index - b.index)
      .map((item) => {
        if (selectedInfo === item.index) {
          return (
            <span
              className={styles.containerSelected}
              onClick={() => this.onInfoOptionUpdate(item.index)}
              key={item.index}
            >
              {item.title}
            </span>
          );
        }

        if (item.index === 2 || item.index === 3) {
          return (
            <span
              className={styles.containerDisabled}
              key={item.index}
            >
              {item.title}
            </span>
          );
        }

        return (
          <span
            className={styles.container}
            onClick={() => this.onInfoOptionUpdate(item.index)}
            key={item.index}
          >
            {item.title}
          </span>
        );
      });
    const infoItems = [tweetsInvolved, accountsInvolved, map, map, diffusionNetwork];

    return (
      <div className={styles.page}>
        <div className={styles.main}>
          <div className={styles.content}>
            { trend }
            <div className={styles.nav_section}>
              <div className={styles.optionContainer}>
                <div className={styles.information}>
                  <div className={styles.listContainer}>{infoOptions}</div>
                </div>
              </div>
            </div>
            {infoItems[selectedInfo]}
          </div>
        </div>
      </div>
    );
  }
}

DetailPage.propTypes = {
  location: PropTypes.object.isRequired,
  getEvent: PropTypes.func.isRequired,
  getProgData: PropTypes.func.isRequired,
  getDiffusionData: PropTypes.func.isRequired,
  getSearchData: PropTypes.func.isRequired,
  isLoadingEvent: PropTypes.bool.isRequired,
  isLoadingProgData: PropTypes.bool.isRequired,
  // isLoadingDiffusionData: PropTypes.bool.isRequired,
  currentEvent: PropTypes.object,
  progData: PropTypes.array,
  diffusionData: PropTypes.array,
  diffusionIds: PropTypes.array,
  userInfo: PropTypes.object.isRequired,
  // tweetInfo: PropTypes.object.isRequired,
  isFromSearch: PropTypes.bool,
};

DetailPage.defaultProps = {
  currentEvent: null,
  // geoInfo: null,
  progData: [],
  diffusionData: [],
  diffusionIds: [],
  isFromSearch: false,
};

export default withRouter(connect(
  state => ({
    // geoInfo: state.data.geoInfo,
    currentEvent: state.details.currentEvent,
    progData: state.details.progData,
    diffusionIds: state.details.diffusionIds,
    diffusionData: state.details.diffusionData,
    isLoadingEvent: state.details.isLoadingEvent,
    isLoadingProgData: state.details.isLoadingProgData,
    // isLoadingDiffusionData: state.details.isLoadingDiffusionData,
    userInfo: state.userInfo,
    tweetInfo: state.tweetInfo,
  }),
  dispatch => ({
    getEvent: eventName => dispatch(fetchEventByName(eventName)),
    getProgData: eventName => dispatch(fetchProgressiveDataByName(eventName)),
    getDiffusionData: ids => dispatch(fetchDiffusionData(ids)),
    getUsers: ids => dispatch(fetchUsersByIds(ids)),
    getTweets: ids => dispatch(fetchTweetsByIds(ids)),
    getSearchData: query => dispatch(fetchSearchData(query)),
  }),
)(DetailPage));
