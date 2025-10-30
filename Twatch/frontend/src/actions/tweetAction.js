import * as types from '../reducers/types';

const fetchTweetsStarted = ids => ({
  type: types.FETCH_TWEET_STARTED,
  payload: {
    ids,
  },
});

const fetchTweetsFailure = ids => ({
  type: types.FETCH_TWEET_FAILURE,
  payload: {
    ids,
  },
});

export const fetchTweetsSuccess = data => ({
  type: types.FETCH_TWEET_SUCCESS,
  payload: {
    data,
  },
});

// eslint-disable-next-line import/prefer-default-export
export const fetchTweetsByIds = ids => (dispatch) => {
  dispatch(fetchTweetsStarted(ids));
  fetch(`${process.env.REACT_APP_API_BASE_URL}tweets?id=${ids.join(',')}`)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }
      throw Error(response.statusText);
    })
    .then((data) => {
      dispatch(fetchTweetsSuccess(data));
    }).catch(() => {
      dispatch(fetchTweetsFailure(ids));
    });
};
