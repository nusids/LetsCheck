import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { withRouter } from 'react-router';
import { connect } from 'react-redux';
import Loader from 'react-loader-spinner';
import InteractiveTrend from '../components/InteractiveTrend';
import { fetchOverallTrend } from '../actions/dataAction';
import styles from './OverallTrendPage.scss';

class OverallTrendPage extends Component {
  constructor() {
    super();

    this.state = {
      currentTime: -1,
      isTimeSelected: false,
    };

    this.xyDataProcessed = false;
    this.xyMap = new Map();
    this.onCurrentTimeChanged = this.onCurrentTimeChanged.bind(this);
  }

  componentDidMount() {
    const { getOverallTrend } = this.props;
    getOverallTrend();
  }

  componentDidUpdate() {
    const { overallTrend } = this.props;
    if (overallTrend && !this.xyDataProcessed) {
      overallTrend.forEach((item) => {
        this.xyMap.set(item.x, item.y);
      });
      this.xyDataProcessed = true;
    }
  }

  onCurrentTimeChanged(time, isSelected) {
    this.setState({
      currentTime: time,
      isTimeSelected: isSelected || false,
    });
  }

  render() {
    const { overallTrend, isLoadingOverallTrend } = this.props;
    const {
      currentTime,
      isTimeSelected,
    } = this.state;

    if (isLoadingOverallTrend) {
      return (
        <div className={styles.page}>
            <div className={styles.loader}>
                <Loader type="RevolvingDot" color="#000000" height="50" />
            </div>
            <span className={styles.message}>Loading trend...</span>
        </div>
      );
    }

    return (
      <div className={styles.main}>
        <h4 className={styles.title}>Total number of tweets over time</h4>
        <div className={styles.chart}>
            <InteractiveTrend
              data={overallTrend}
              currentTimeValue={{
                x: currentTime,
                y: this.xyMap.get(currentTime),
              }}
              isTimeSelected={isTimeSelected}
              onChange={this.onCurrentTimeChanged}
              onRangeChange={() => {}}
            />
        </div>
      </div>
    );
  }
}

OverallTrendPage.propTypes = {
  getOverallTrend: PropTypes.func.isRequired,
  overallTrend: PropTypes.array,
  isLoadingOverallTrend: PropTypes.bool,
};

OverallTrendPage.defaultProps = {
  overallTrend: [],
  isLoadingOverallTrend: false,
};

export default withRouter(connect(
  state => ({
    isLoadingOverallTrend: state.overallData.isLoadingOverallTrend,
    overallTrend: state.overallData.overallTrend,
  }),
  dispatch => ({
    getOverallTrend: () => dispatch(fetchOverallTrend()),
  }),
)(OverallTrendPage));
