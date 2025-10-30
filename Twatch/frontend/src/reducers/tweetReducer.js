/* eslint-disable no-case-declarations */
import * as types from './types';

const initalState = {};
let temp = initalState;

const tweetInfo = (state = initalState, action) => {
  switch (action.type) {
    case types.FETCH_TWEET_STARTED:
      temp = { ...state };
      action.payload.ids.forEach((id) => {
        temp[id] = {
          isLoading: true,
          isValidTweet: false,
        };
      });
      return temp;
    case types.FETCH_TWEET_FAILURE:
      temp = { ...state };
      action.payload.ids.forEach((id) => {
        temp[id] = {
          isLoading: false,
          isValidTweet: false,
        };
      });
      return temp;
    case types.FETCH_TWEET_SUCCESS:
      temp = { ...state };
      action.payload.data.forEach((item) => {
        temp[item.tweet_id] = {
          ...item,
          isLoading: false,
          isValidTweet: true,
        };
      });
      return temp;
    default:
      return state;
  }
};

export default tweetInfo;
