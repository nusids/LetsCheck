import * as types from './types';

const initalState = {
  isLoadingEvent: true,
  isLoadingProgData: true,
  isLoadingDiffusionIds: false,
  isLoadingDiffusionData: true,
  currentEvent: null,
  progData: [],
  diffusionData: [],
  diffusionIds: [],
  urlInfo: {},
};

const details = (state = initalState, action) => {
  switch (action.type) {
    case types.FETCH_EVENT_BY_NAME_STARTED:
      return {
        ...state,
        isLoadingEvent: true,
        currentEvent: null,
      };
    case types.FETCH_EVENT_BY_NAME_FAILURE:
      return {
        ...state,
        isLoadingEvent: false,
        currentEvent: null,
      };
    case types.FETCH_EVENT_BY_NAME_SUCCESS:
      return {
        ...state,
        isLoadingEvent: false,
        currentEvent: action.payload.data,
      };
    case types.FETCH_PROG_DATA_BY_NAME_STARTED:
      return {
        ...state,
        isLoadingProgData: true,
        progData: [],
      };
    case types.FETCH_PROG_DATA_BY_NAME_FAILURE:
      return {
        ...state,
        isLoadingProgData: false,
        progData: [],
      };
    case types.FETCH_PROG_DATA_BY_NAME_SUCCESS:
      return {
        ...state,
        isLoadingProgData: false,
        progData: action.payload.data,
      };
    case types.FETCH_DIFFUSION_DATA_STARTED:
      return {
        ...state,
        isLoadingDiffusionData: true,
        diffusionData: [],
      };
    case types.FETCH_DIFFUSION_DATA_FAILURE:
      return {
        ...state,
        isLoadingDiffusionData: false,
        diffusionData: [],
      };
    case types.FETCH_DIFFUSION_DATA_SUCCESS:
      return {
        ...state,
        isLoadingDiffusionData: false,
        diffusionData: action.payload.data,
      };
    case types.FETCH_DIFFUSION_IDS_STARTED:
      return {
        ...state,
        // isLoadingDiffusionData: true,
        diffusionIds: [],
      };
    case types.FETCH_DIFFUSION_IDS_FAILURE:
      return {
        ...state,
        // isLoadingDiffusionData: false,
        diffusionIds: [],
      };
    case types.FETCH_DIFFUSION_IDS_SUCCESS:
      return {
        ...state,
        // isLoadingDiffusionData: false,
        diffusionIds: action.payload.data,
      };
    default:
      return state;
  }
};

export default details;
