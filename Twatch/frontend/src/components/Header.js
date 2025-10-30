import React, { Component } from 'react';
import PropTypes from 'prop-types';
import * as styles from './Header.scss';
import logoImg from '../assets/logo.svg';

class Header extends Component {
  constructor(props) {
    super(props);

    this.onLogoClicked = this.onLogoClicked.bind(this);
  }

  onLogoClicked() {
    const { history } = this.props;
    history.push('/');
  }

  render() {
    const { isFixed } = this.props;
    return (
      <div className={isFixed ? styles.headerFixed : styles.header}>
        <img className={styles.logo} src={logoImg} onClick={this.onLogoClicked} alt="Twatch" />
        {/* <div className={styles.signInButton}>
          <img className={styles.buttonIcon} src={twitterImg} alt="" />
            Sign In
        </div> */}
      </div>
    );
  }
}

Header.propTypes = {
  history: PropTypes.object.isRequired,
  isFixed: PropTypes.bool,
};

Header.defaultProps = {
  isFixed: false,
};

export default Header;
