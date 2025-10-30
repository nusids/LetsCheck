import { combineReducers } from 'redux';
import userInfo from './userInfoReducer';
import details from './detailReducer';
import tweetInfo from './tweetReducer';
import overallData from './dataReducer';

export default combineReducers({
  userInfo, details, tweetInfo, overallData,
});
