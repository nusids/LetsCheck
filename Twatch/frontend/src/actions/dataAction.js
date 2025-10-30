import * as types from '../reducers/types';

const fetchEventGeoInfoStarted = () => ({
  type: types.FETCH_EVENT_GEOINFO_STARTED,
});

const fetchEventGeoInfoFailure = () => ({
  type: types.FETCH_EVENT_GEOINFO_FAILURE,
});

const fetchEventGeoInfoSuccess = (eventName, items) => ({
  type: types.FETCH_EVENT_GEOINFO_SUCCESS,
  payload: {
    eventName,
    items,
  },
});

const fetchOverallTrendStarted = () => ({
  type: types.FETCH_OVERALL_TREND_STARTED,
});

const fetchOverallTrendFailure = () => ({
  type: types.FETCH_OVERALL_TREND_FAILURE,
});

const fetchOverallTrendSuccess = data => ({
  type: types.FETCH_OVERALL_TREND_SUCCESS,
  payload: {
    data,
  },
});

// eslint-disable-next-line import/prefer-default-export
export const fetchEventGeoInfo = eventName => (dispatch) => {
  dispatch(fetchEventGeoInfoStarted());

  fetch(`${process.env.REACT_APP_API_BASE_URL}geoInfo?event=${eventName}`)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }

      dispatch(fetchEventGeoInfoFailure());
      return false;
    })
    .then((items) => {
      dispatch(fetchEventGeoInfoSuccess(eventName, items));
    });
};

export const fetchOverallTrend = () => (dispatch) => {
  dispatch(fetchOverallTrendStarted());
  fetch(`${process.env.REACT_APP_API_BASE_URL}overallTrend`)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }
      throw Error(response.statusText);
    })
    .then((data) => {
      // eslint-disable-next-line no-console
      dispatch(fetchOverallTrendSuccess(data));
    }).catch(() => {
      dispatch(fetchOverallTrendFailure());
    });
};
