import * as types from '../reducers/types';
import 'react-vis/dist/style.css';

export const fetchEventByNameStarted = () => ({
  type: types.FETCH_EVENT_BY_NAME_STARTED,
});

export const fetchEventByNameFailure = () => ({
  type: types.FETCH_EVENT_BY_NAME_FAILURE,
});

export const fetchEventByNameSuccess = data => ({
  type: types.FETCH_EVENT_BY_NAME_SUCCESS,
  payload: {
    data,
  },
});

const fetchProgDataByNameStarted = () => ({
  type: types.FETCH_PROG_DATA_BY_NAME_STARTED,
});

const fetchProgDataByNameFailure = () => ({
  type: types.FETCH_PROG_DATA_BY_NAME_FAILURE,
});

export const fetchProgDataByNameSuccess = data => ({
  type: types.FETCH_PROG_DATA_BY_NAME_SUCCESS,
  payload: {
    data,
  },
});

const fetchDiffusionDataStarted = () => ({
  type: types.FETCH_DIFFUSION_DATA_STARTED,
});

const fetchDiffusionDataFailure = () => ({
  type: types.FETCH_DIFFUSION_DATA_FAILURE,
});

export const fetchDiffusionDataSuccess = data => ({
  type: types.FETCH_DIFFUSION_DATA_SUCCESS,
  payload: {
    data,
  },
});

export const fetchDiffusionIdsSuccess = data => ({
  type: types.FETCH_DIFFUSION_IDS_SUCCESS,
  payload: {
    data,
  },
});

export const fetchEventByName = eventName => (dispatch) => {
  dispatch(fetchEventByNameStarted());

  fetch(`${process.env.REACT_APP_API_BASE_URL}event?event=${eventName}`)
    .then((response) => {
      if (response.ok) {
        // return response.json();
      }
      dispatch(fetchEventByNameFailure());
      return false;
    })
    .then((data) => {
      dispatch(fetchEventByNameSuccess(data));
    });
};

export const fetchProgressiveDataByName = eventName => (dispatch) => {
  dispatch(fetchProgDataByNameStarted());

  fetch(`${process.env.REACT_APP_API_BASE_URL}progData?event=${eventName}`)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }
      dispatch(fetchProgDataByNameFailure());
      return false;
    })
    .then((data) => {
      dispatch(fetchProgDataByNameSuccess(data));
    });
};

export const fetchDiffusionData = ids => (dispatch) => {
  dispatch(fetchDiffusionDataStarted());
  // Using POST to prevent 400 response due to long url
  const requestOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: ids }),
  };
  fetch(`${process.env.REACT_APP_API_BASE_URL}diffusionData`, requestOptions)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }
      dispatch(fetchDiffusionDataFailure());
      return false;
    })
    .then((data) => {
      dispatch(fetchDiffusionDataSuccess(data));
    });
};
