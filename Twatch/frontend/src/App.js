import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import HomePage from './pages/HomePage';
import DetailPage from './pages/DetailPage';
import styles from './App.scss';
import OverallTrendPage from './pages/OverallTrendPage';

const SearchPage = props => (
  <DetailPage
    isFromSearch
    {...props}
  />
);

const App = () => (
  <div className={styles.wrapper}>
    <Router basename={process.env.BASENAME ? process.env.BASENAME : '/twatch'}>
      <Switch>
        <Route path="/details" component={DetailPage} />
        <Route path="/search" component={SearchPage} />
        <Route path="/trend" component={OverallTrendPage} />
        <Route component={HomePage} />
      </Switch>
    </Router>
  </div>
);

export default App;
