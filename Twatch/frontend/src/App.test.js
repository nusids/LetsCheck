/* eslint-env jest */

import React from 'react';
import { Provider } from 'react-redux';
import renderer from 'react-test-renderer';
import configureStore from 'redux-mock-store';
import App from './App';

const mockStore = configureStore([]);

describe('My Connected React-Redux Component', () => {
  let store;
  let component;

  beforeEach(() => {
    store = mockStore({
      myState: 'testing',
    });

    component = renderer.create(
      <Provider store={store}>
        <App />
      </Provider>,
    );
  });

  it('renders without crashing', () => {
    console.log(component);
  });
});
