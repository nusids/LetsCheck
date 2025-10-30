import * as types from '../reducers/types';

const fetchUserInfoStarted = screenName => ({
  type: types.FETCH_USER_INFO_STARTED,
  payload: {
    screenName,
  },
});

const fetchUserInfoFailure = screenName => ({
  type: types.FETCH_USER_INFO_FAILURE,
  payload: {
    screenName,
  },
});

const fetchUserInfoSuccess = (screenName, info) => ({
  type: types.FETCH_USER_INFO_SUCCESS,
  payload: {
    screenName,
    info,
  },
});

const fetchUsersBatchStarted = ids => ({
  type: types.FETCH_USER_BATCH_STARTED,
  payload: {
    ids,
  },
});

const fetchUsersBatchFailure = ids => ({
  type: types.FETCH_USER_BATCH_FAILURE,
  payload: {
    ids,
  },
});

export const fetchUsersBatchSuccess = (data, invalidIds) => ({
  type: types.FETCH_USER_BATCH_SUCCESS,
  payload: {
    data,
    invalidIds,
  },
});

// eslint-disable-next-line import/prefer-default-export
export const fetchUserInfoByScreenName = screenName => (dispatch) => {
  dispatch(fetchUserInfoStarted(screenName));

  fetch(`${process.env.REACT_APP_API_BASE_URL}twitter/user?screen_name=${screenName}`)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }

      throw Error(response.statusText);
    })
    .then((info) => {
      dispatch(fetchUserInfoSuccess(screenName, info));
    }).catch(() => {
      dispatch(fetchUserInfoFailure(screenName));
    });
};

export const fetchUsersByIds = ids => (dispatch) => {
  dispatch(fetchUsersBatchStarted(ids));

  fetch(`${process.env.REACT_APP_API_BASE_URL}twitter/users?id=${ids.join(',')}`)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }

      throw Error(response.statusText);
    })
    .then((data) => {
      const validIdSet = new Set(data.map(item => item.id_str));
      const invalidIds = ids.filter(x => !validIdSet.has(x));
      dispatch(fetchUsersBatchSuccess(data, invalidIds));
    }).catch(() => {
      dispatch(fetchUsersBatchFailure(ids));
    });
};
