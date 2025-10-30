/* eslint-disable no-unreachable */
import { fetchTweetsSuccess } from './tweetAction';
import { fetchUsersBatchSuccess } from './userAction';
import {
  fetchDiffusionIdsSuccess,
  fetchProgDataByNameSuccess,
  fetchEventByNameSuccess,
  fetchEventByNameStarted,
  fetchEventByNameFailure,
  fetchDiffusionDataSuccess,
} from './detailsAction';

// eslint-disable-next-line import/prefer-default-export
export const fetchSearchData = query => (dispatch) => {
  dispatch(fetchEventByNameStarted());

  fetch(`${process.env.REACT_APP_API_BASE_URL}search?q=${encodeURIComponent(query)}`)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }
      dispatch(fetchEventByNameFailure());
      return response.status;
    })
    .then((data) => {
      dispatch(fetchEventByNameSuccess(data.event));
      dispatch(fetchUsersBatchSuccess(data.userInfo || [], []));
      dispatch(fetchTweetsSuccess(data.tweetInfo || []));
      dispatch(fetchDiffusionIdsSuccess(data.diffusionIds || []));
      dispatch(fetchDiffusionDataSuccess([]));
      dispatch(fetchProgDataByNameSuccess(data.progData || []));
    });
};
