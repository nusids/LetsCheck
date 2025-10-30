import React, { Component } from 'react';
import PropTypes from 'prop-types';
import {
  FlexibleWidthXYPlot, XAxis, YAxis, LineSeries, Hint, Highlight,
} from 'react-vis';
import moment from 'moment';
import styles from './InteractiveTrend.scss';

class InteractiveTrend extends Component {
  constructor(props) {
    super(props);

    this.state = {
      mouseValue: { x: -1, y: 0 },
    };

    this.currentMouseValue = { x: -1, y: 0 };
    this.onMouseOutTrend = this.onMouseOutTrend.bind(this);
    this.onMouseClickTrend = this.onMouseClickTrend.bind(this);
  }

  onMouseOverTrend(value) {
    this.currentMouseValue = value;

    const { onChange, isTimeSelected } = this.props;

    if (!isTimeSelected) {
      onChange(this.currentMouseValue.x);
    }
  }

  onMouseOutTrend() {
    const { onChange, isTimeSelected } = this.props;

    if (!isTimeSelected) {
      onChange(-1);
    }
  }

  onMouseClickTrend() {
    const { onChange } = this.props;

    onChange(this.currentMouseValue.x, true);
  }

  onSelectedAreaUpdate(area) {
    const { onRangeChange } = this.props;
    onRangeChange(area.left, area.right);
  }

  render() {
    const {
      data, currentTimeValue, isTimeSelected, isLog,
    } = this.props;
    const { mouseValue } = this.state;
    const shouldShowHint = currentTimeValue.x > 0 || isTimeSelected;

    const trend = (
      <FlexibleWidthXYPlot
        height={300}
        margin={{
          left: 50, right: 10, top: 10, bottom: 40,
        }}
        animation
        onClick={this.onMouseClickTrend}
        onMouseLeave={this.onMouseOutTrend}
        style={{
          border: 'solid #dedede 1px',
        }}
      >
        <XAxis
          hideLine
          tickSize={0}
          tickTotal={6}
          tickFormat={v => moment.unix(v).format('MMM DD, YYYY')}
          style={{
            text: { fontSize: '0.8rem' },
          }}
        />
        <YAxis
          hideLine
          tickTotal={4}
          tickSize={0}
          tickFormat={val => (Math.round(val) === val ? val : '')}
          yType={isLog ? 'log' : 'linear'}
          style={{
            text: { fontSize: '0.8rem' },
          }}
        />
        <LineSeries
          curve="curveBasis"
          color="#6A93B1"
          data={data}
          yType={isLog ? 'log' : 'linear'}
          animation
          onNearestX={value => this.onMouseOverTrend(value)}
        />
        {shouldShowHint ? (
          <Hint
            value={{ x: currentTimeValue.x, y: 0 }}
            align={{ vertical: 'center', horizontal: 'right' }}
          >
            <div className={styles.hintContainer}>
              <div className={styles.textContainer}>
                <div className={styles.number}>{`${currentTimeValue.x > 0 ? currentTimeValue.y : mouseValue.y} tweets on ${moment.unix(currentTimeValue.x).format('MMM DD')}`}</div>
              </div>
              <div className={styles.indicator} />
            </div>
          </Hint>
        ) : null}
        {shouldShowHint && !isLog ? (
          <Hint
            value={{ x: currentTimeValue.x, y: currentTimeValue.y }}
            align={{ vertical: 'top', horizontal: 'right' }}
          >
            <div className={styles.dot} />
          </Hint>
        ) : null}
        <Highlight
          drag
          enableY={false}
          onBrush={area => this.onSelectedAreaUpdate(area)}
          onDrag={area => this.onSelectedAreaUpdate(area)}
        />
      </FlexibleWidthXYPlot>
    );

    return trend;
  }
}

InteractiveTrend.propTypes = {
  data: PropTypes.array.isRequired,
  currentTimeValue: PropTypes.object.isRequired,
  isTimeSelected: PropTypes.bool.isRequired,
  isLog: PropTypes.bool,
  onChange: PropTypes.func.isRequired,
  onRangeChange: PropTypes.func.isRequired,
};

InteractiveTrend.defaultProps = {
  isLog: false,
};

export default InteractiveTrend;
