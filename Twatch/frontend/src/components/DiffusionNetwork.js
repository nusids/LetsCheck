import React, { Component } from 'react';
import PropTypes from 'prop-types';
import _ from 'underscore';
import * as numeral from 'numeral';
// import { ContinuousSizeLegend } from 'react-vis';
import { InteractiveForceGraph, ForceGraphNode, ForceGraphLink } from 'react-vis-force';
import Loader from 'react-loader-spinner';
import DiffusionNetworkDetails from './DiffusionNetworkDetails';
import * as ColorUtils from '../utils/colorUtils';
import * as styles from './DiffusionNetwork.scss';

const ReplyType = Object.freeze({
  agreed: 'agreed',
  disagreed: 'disagreed',
  appealForMoreInfo: 'appeal-for-more-information',
  comment: 'comment',
});

const positiveHex = '#019EFF';
const negativeHex = '#FF1010';
const appealForMoreInfoHex = '#BA52FF';
const neutralHex = '#989898';
const deepNeutralHex = '#545c61';
const retweetHex = '#F29B1D';

const getColorItems = (isInCluster) => {
  if (isInCluster) {
    return [
      {
        color: deepNeutralHex,
        title: 'Source',
      },
      {
        color: neutralHex,
        title: 'Comment',
      },
      {
        color: retweetHex,
        title: 'Retweets',
      },
    ];
  }
  return [];
};

const getFillByAnnotation = (annotation) => {
  const {
    compound, misinformation, isRetweet, isReply,
  } = annotation;
  const positiveRGB = ColorUtils.hexToRgb(positiveHex);
  const negativeRGB = ColorUtils.hexToRgb(negativeHex);
  const white = ColorUtils.hexToRgb('#FFFFFF');

  if (isRetweet) {
    return retweetHex;
  }

  if (isReply) {
    const type = annotation.responsetype_vs_source;

    switch (type) {
      case ReplyType.agreed:
        return positiveHex;
      case ReplyType.disagreed:
        return negativeHex;
      case ReplyType.appealForMoreInfo:
        return appealForMoreInfoHex;
      default:
        return neutralHex;
    }
  }

  if (compound === undefined) {
    // if (annotation.is_turnaround === 1) return turnaroundHex;
    if (annotation.proven_true === 1) return positiveHex;
    if (misinformation === 1) return negativeHex;
    return deepNeutralHex;
  }

  const positive = 0.05;
  const negative = -0.05;
  const opacityThreshold = 0.3;
  let percentage = 1; // percentage of own color, not white

  if (compound >= positive) {
    percentage = (compound - positive) * (1 - opacityThreshold) + opacityThreshold;
    const mixed = ColorUtils.colorMixer(positiveRGB, white, percentage);
    return ColorUtils.rgbToHex(mixed.r, mixed.g, mixed.b);
  } if (compound <= negative) {
    percentage = (-compound + negative) * (1 - opacityThreshold) + opacityThreshold;
    const mixed = ColorUtils.colorMixer(negativeRGB, white, percentage);
    return ColorUtils.rgbToHex(mixed.r, mixed.g, mixed.b);
  }

  return neutralHex;
};

const getLogResult = (number) => {
  if (number <= 0) {
    return 0;
  }
  return Math.max(1, Math.log2(number));
};

class DiffusionNetwork extends Component {
  constructor(props) {
    super(props);

    this.state = {
      selectedNode: null,
      isInCluster: false,
      mouseOverNode: null,
      sourceTweetNode: null,
      isLoading: true,
    };

    this.width = 400;
    this.powerSum = 0;
    this.links = [];
    this.nodes = [];
    this.influentialAccountIdsInCluster = [];
    this.smallestCircleSize = 0;
    this.largestCircleSize = 0;
    this.smallestOriginalSize = 0;
    this.largestOriginalSize = 0;
    this.numTweets = 0;
    this.allTweets = [];
    this.onNodeSelected = this.onNodeSelected.bind(this);
    this.onNodeDeselected = this.onNodeDeselected.bind(this);
    this.onCloseButtonClicked = this.onCloseButtonClicked.bind(this);
    this.onRedirectButtonClicked = this.onRedirectButtonClicked.bind(this);
    this.onMouseOverNode = this.onMouseOverNode.bind(this);
    this.onMouseLeaveNode = this.onMouseLeaveNode.bind(this);
    this.onMouseEnterGraph = this.onMouseEnterGraph.bind(this);
    this.onMouseLeaveGraph = this.onMouseLeaveGraph.bind(this);
  }

  componentDidMount() {
    this.updateData();
  }

  componentDidUpdate(prevProps, prevState) {
    const prevTime = prevProps.currentTime;
    const prevSourceNode = prevState.sourceTweetNode;
    const { currentTime } = this.props;
    const { sourceTweetNode, isLoading } = this.state;

    if ((isLoading || sourceTweetNode) && (prevTime !== currentTime || (this.nodes.length === 0 && currentTime < 0) || prevSourceNode !== sourceTweetNode)) {
      this.updateData();
    }
  }

  onMouseOverNode(node) {
    this.setState({
      mouseOverNode: node,
    });
  }

  onMouseLeaveNode() {
    this.setState({
      mouseOverNode: null,
    });
  }

  // eslint-disable-next-line no-unused-vars
  onNodeSelected(e, node) {
    const { isInCluster } = this.state;

    if (isInCluster) {
      this.setState({
        selectedNode: node,
        mouseOverNode: null,
      });
    } else {
      this.setState({
        sourceTweetNode: node,
        isInCluster: true,
        mouseOverNode: null,
      });
    }
  }

  // eslint-disable-next-line no-unused-vars
  onNodeDeselected(e, node) {
    this.setState({
      selectedNode: null,
    });
  }

  onCloseButtonClicked() {
    this.setState({
      selectedNode: null,
      mouseOverNode: null,
      sourceTweetNode: 'all',
      isInCluster: false,
    });
  }

  onRedirectButtonClicked() {
    const { sourceTweetNode } = this.state;
    const { onRedirectToTIBClicked } = this.props;

    onRedirectToTIBClicked(sourceTweetNode.id_str);
  }

  onMouseEnterGraph() {
    document.body.style.overflow = 'hidden';
  }

  onMouseLeaveGraph() {
    document.body.style.overflow = 'unset';
  }

  updateData() {
    const { currentTime, data } = this.props;
    const { sourceTweetNode } = this.state;

    if (!data || data.length === 0) {
      return;
    }

    if (!(_.find(data, d => d.tweet_id))) {
      this.setState({ isLoading: false });
      return;
    }

    this.links = [];
    this.nodes = [];
    this.numTweets = 0;

    // If powerSum has not been calculated
    if (this.powerSum <= 0) {
      data.forEach((item) => {
        if (item) {
          const size = item.num_retweets + item.replies.length;
          const logResult = getLogResult(size);
          this.powerSum += (logResult * logResult);
        }
      });
    }

    if (!sourceTweetNode || sourceTweetNode === 'all') {
      // Plot main diffusion network
      data.forEach((item) => {
        if (!item.tweet_id) {
          return;
        }
        const filteredReplies = item.replies.filter(x => x.unix <= currentTime || currentTime < 0);
        const size = item.num_retweets + filteredReplies.length + 1;

        if (size > 0) {
          const modifiedSize = getLogResult(size) * 3.14;
          this.numTweets += size;

          this.nodes.push({
            id_str: item.tweet_id,
            size: modifiedSize,
            originalSize: size,
            num_retweets_shown: item.num_retweets,
            num_replies_shown: filteredReplies.length,
            annotation: item.annotation,
            data: _.omit(item, ['annotation', 'replies', 'reply_structure']),
            label: `${size} ${size === 1 ? 'tweet' : 'tweets'}`,
          });
        }
      });
    } else {
      // Plot zoomed in diffusion network
      this.allTweets = [];
      this.influentialAccountIdsInCluster = [];
      const dataOfSourceNode = data.find(x => x.tweet_id === sourceTweetNode.id_str);

      if (dataOfSourceNode.unix <= currentTime || currentTime < 0) {
        const filteredReplies = dataOfSourceNode.replies.filter(x => x.unix <= currentTime || currentTime < 0);

        const all = _.uniq(filteredReplies.concat([dataOfSourceNode]), 'user_id');
        all.sort((a, b) => b.user_followers_count - a.user_followers_count);
        this.influentialAccountIdsInCluster = all.slice(0, 6).map(item => ({
          id_str: item.user_id,
          followers_count: item.user_followers_count,
        }));
        this.numTweets = dataOfSourceNode.num_retweets + filteredReplies.length + 1;
        // this.allTweets.push(...filteredChildren.map(item => ({
        //   ..._.pick(item, ['tweet_id', 'unix']),
        //   isRetweet: true,
        // })));
        this.allTweets.push(...filteredReplies.map(item => _.pick(item, ['tweet_id', 'unix'])));
        this.allTweets.push(_.pick(dataOfSourceNode, ['tweet_id', 'unix']));

        this.nodes.push({
          id_str: dataOfSourceNode.tweet_id,
          size: getLogResult(dataOfSourceNode.user_followers_count),
          originalSize: dataOfSourceNode.user_followers_count,
          annotation: {
            ...dataOfSourceNode.annotation,
            isSourceTweet: true,
          },
          data: dataOfSourceNode,
          label: `Source tweet - ${numeral(dataOfSourceNode.user_followers_count).format('0a')} followers`,
        });
        const childrenSize = dataOfSourceNode.num_retweets;
        const modifiedSize = getLogResult(childrenSize);
        this.nodes.push({
          id_str: 'retweets',
          size: modifiedSize,
          originalSize: childrenSize,
          annotation: {
            isRetweet: true,
            count: childrenSize,
          },
          label: `${childrenSize} Retweets`,
        });
        this.nodes.push(...filteredReplies.map(x => ({
          id_str: x.tweet_id,
          size: getLogResult(x.user_followers_count),
          originalSize: x.user_followers_count,
          label: `Tweet reply - ${numeral(x.user_followers_count).format('0a')} followers`,
          data: x,
          annotation: {
            ...x.sentiment,
            isReply: true,
          },
        })));

        this.links.push(...filteredReplies.map(x => ({
          source: dataOfSourceNode.tweet_id,
          target: x.tweet_id,
        })));
      }
    }

    const allSizes = this.nodes.map(item => item.size);
    const allOriginalSizes = this.nodes.map(item => item.originalSize);
    this.smallestOriginalSize = Math.min(...allOriginalSizes);
    this.largestOriginalSize = Math.max(...allOriginalSizes);
    this.smallestCircleSize = Math.min(...allSizes);
    this.largestCircleSize = Math.max(...allSizes);
    this.setState({ isLoading: false });
    this.forceUpdate();
  }

  render() {
    const {
      selectedNode, isInCluster, mouseOverNode, sourceTweetNode, isLoading,
    } = this.state;
    if (isLoading) {
      return (<div className={styles.messageContainer}>
                <div className={styles.loader}>
                  <Loader type="RevolvingDot" color="#000000" height="50" />
                </div>
                <br />
                <span className={styles.message}>Building diffusion network...</span>
              </div>);
    }

    if (this.nodes.length === 0) {
      return (<div className={styles.errorMessage}>
        <span>Error building diffusion network, please select another time on the timeline.</span>
              </div>);
    }

    const { currentTime, eventName } = this.props;
    const colorLegendElements = getColorItems(isInCluster).map(item => (
        <div className={styles.colorItem} key={item.color}>
          <div className={styles.circle} style={{ backgroundColor: item.color }} />
          <div className={styles.desc}>{item.title}</div>
        </div>
    ));
    const linkElements = this.links.map(link => (
      <ForceGraphLink
        link={{
          source: link.source,
          target: link.target,
          value: 1, // strokeWidth
        }}
        key={link.target}
      />
    ));
    const nodeElements = this.nodes.map((node) => {
      const colorResult = getFillByAnnotation(node.annotation);

      return (
        <ForceGraphNode
          node={{
            id: node.id_str,
            id_str: node.id_str,
            radius: node.size,
            annotation: node.annotation,
            label: node.label,
            data: node.data,
          }}
          key={node.id_str}
          fill={colorResult}
          onMouseOver={() => this.onMouseOverNode(node)}
          onMouseLeave={this.onMouseLeaveNode}
          strokeWidth={node.annotation.isSourceTweet ? 2 : 1}
          stroke={node.annotation.isSourceTweet ? '#646464' : '#FFF'}
          labelStyle={{ fontSize: '0.9rem' }}
        />
      );
    });

    return (
      <div className={styles.container}>
        {!isInCluster || (
            <div className={styles.detailHeaderContainer}>
              <div className={styles.actionContainer}>
                <div className={styles.actionBackContainer} onClick={this.onCloseButtonClicked}>
                  &lt; Back to main view
                </div>
              </div>
              <div className={styles.legendContainer}>
                <div className={styles.colorLegend}>{colorLegendElements}</div>
              </div>
            </div>
        )}
        <div className={styles.contentContainer}>
          <div className={styles.chartContainer}>
            <div
              className={styles.graphContainer}
              onMouseEnter={this.onMouseEnterGraph}
              onMouseLeave={this.onMouseLeaveGraph}
            >
              <InteractiveForceGraph
                zoom
                labelOffset={{
                  x: ({ radius }) => -radius,
                  y: ({ radius }) => -radius - 5,
                }}
                labelAttr="label"
                simulationOptions={{
                  animate: true,
                  width: this.width,
                  height: this.width,
                  strength: {
                  // collide: 0,
                    charge: 1,
                  },
                  alphaDecay: 0.066, // 50 ticks for 0.129, 100 for 0.066, 200 for 0.033
                  radiusMargin: 2,
                }}
                onSelectNode={this.onNodeSelected}
                onDeselectNode={this.onNodeDeselected}
                key={isInCluster ? `1-${currentTime}` : `0-${currentTime}`}
              >
                {linkElements}
                {nodeElements}
              </InteractiveForceGraph>
            </div>
          </div>
          <div className={styles.detailsContainer}>
            <DiffusionNetworkDetails
              eventName={eventName}
              isInCluster={isInCluster}
              sourceTweet={sourceTweetNode}
              mouseoverTweet={mouseOverNode}
              selectedTweet={selectedNode}
              influentialAccountIds={this.influentialAccountIdsInCluster}
              numNodes={nodeElements.length}
              numTweets={this.numTweets}
              allTweets={this.allTweets}
            />
          </div>
        </div>
      </div>
    );
  }
}

DiffusionNetwork.propTypes = {
  currentTime: PropTypes.number.isRequired,
  data: PropTypes.array.isRequired,
  eventName: PropTypes.string,
  onRedirectToTIBClicked: PropTypes.func.isRequired,
};

DiffusionNetwork.defaultProps = {
  eventName: '',
};

export default DiffusionNetwork;
