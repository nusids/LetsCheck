import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { withRouter } from 'react-router';
import { connect } from 'react-redux';
import searchImg from '../assets/search.svg';
import styles from './HomePage.scss';
import Header from '../components/Header';

class HomePage extends Component {
  constructor(props) {
    super(props);

    this.searchText = '';
    this.onSearch = this.onSearch.bind(this);
    this.onInputChange = this.onInputChange.bind(this);
    this.onInputKeyUp = this.onInputKeyUp.bind(this);
    this.onViewEvents = this.onViewEvents.bind(this);
  }

  onSearch() {
    const textRaw = this.searchText.trim();
    if (!textRaw) {
      return;
    }

    const { history } = this.props;
    history.push(`/search?q=${textRaw}`);
  }

  onInputChange(e) {
    this.searchText = e.target.value.trim();
  }

  onInputKeyUp(e) {
    e.preventDefault();

    if (e.keyCode === 13) {
      this.onSearch();
    }
  }

  onViewEvents() {
    const { history } = this.props;
    history.push('/events');
  }

  render() {
    const { history } = this.props;

    return (
      <div className={styles.page}>
        <Header history={history} />
        <div className={styles.middle}>
          <div className={styles.searchArea}>
            <input
              className={styles.searchInput}
              type="text"
              placeholder="Type to search..."
              onChange={this.onInputChange}
              onKeyUp={this.onInputKeyUp}
            />
            <img className={styles.inputIcon} src={searchImg} alt="" onClick={this.onSearch} />
          </div>
        </div>
      </div>
    );
  }
}

HomePage.propTypes = {
  history: PropTypes.object.isRequired,
};

export default withRouter(connect()(HomePage));
