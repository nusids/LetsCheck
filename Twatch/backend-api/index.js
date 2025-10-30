const express = require('express');
const compression = require('compression');
const bodyParser = require('body-parser');
const port = process.env.PORT || 8080;
const apiRoutes = require('./api');
const helmet = require('helmet')

const runBackend = () => {
  const app = express();
  app.use(function (req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
  });
  app.use(bodyParser.urlencoded({
    limit: "50mb",
    extended: true,
    parameterLimit: Number.MAX_SAFE_INTEGER
  }));
  app.use(bodyParser.json({limit: "50mb"}));

  app.use(compression());
  app.use(helmet());

  app.use('/api', apiRoutes);

  app.get('/', (req, res) => res.send('Twatch backend running'));

  app.listen(port, function () {
    console.log("Running on port " + port);
  });
}

runBackend();
