import * as types from './types';

const initalState = {
  isLoadingOverallTrend: false,
  overallTrend: null,
};

const overallData = (state = initalState, action) => {
  switch (action.type) {
    case types.FETCH_OVERALL_TREND_STARTED:
      return {
        ...state,
        isLoadingOverallTrend: true,
        overallTrend: null,
      };
    case types.FETCH_OVERALL_TREND_FAILURE:
      return {
        ...state,
        isLoadingOverallTrend: false,
        overallTrend: null,
      };
    case types.FETCH_OVERALL_TREND_SUCCESS:
      return {
        ...state,
        isLoadingOverallTrend: false,
        overallTrend: action.payload.data,
      };
    default:
      return state;
  }
};

export default overallData;
