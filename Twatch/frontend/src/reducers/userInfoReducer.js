import * as types from './types';

const initalState = {};
let temp = initalState;

const userInfo = (state = initalState, action) => {
  switch (action.type) {
    case types.FETCH_USER_INFO_STARTED:
      return {
        ...state,
        [action.payload.screenName]: {
          isLoading: true,
          isValidUser: false,
        },
      };
    case types.FETCH_USER_INFO_FAILURE:
      return {
        ...state,
        [action.payload.screenName]: {
          isLoading: false,
          isValidUser: false,
        },
      };
    case types.FETCH_USER_INFO_SUCCESS:
      return {
        ...state,
        [action.payload.screenName]: {
          ...action.payload.info,
          isValidUser: true,
          isLoading: false,
        },
      };
    case types.FETCH_USER_BATCH_STARTED:
      temp = { ...state };
      action.payload.ids.forEach((id) => {
        temp[id] = {
          isLoading: true,
          isValidUser: false,
        };
      });
      return temp;
    case types.FETCH_USER_BATCH_FAILURE:
      temp = { ...state };
      action.payload.ids.forEach((id) => {
        temp[id] = {
          isLoading: false,
          isValidUser: false,
        };
      });
      return temp;
    case types.FETCH_USER_BATCH_SUCCESS:
      temp = { ...state };
      action.payload.data.forEach((item) => {
        try {
          if (item) {
            temp[item.user_id] = {
              ...item,
              isLoading: false,
              isValidUser: true,
            };
          }
        } catch (e) {
          throw e;
        }
      });
      action.payload.invalidIds.forEach((id) => {
        temp[id] = {
          isLoading: false,
          isValidUser: false,
        };
      });
      return temp;
    default:
      return state;
  }
};

export default userInfo;
