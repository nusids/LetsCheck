#!/bin/bash

# clone the repository
git clone https://github.com/dinhnhobao/Twatch
# or extract the initial Twatch folder

# go to the main directory
cd Twatch

# go to backend folder
cd backend

# install MongoDB on Ubuntu
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 68818C72E52529D4
sudo echo "deb http://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org

# start mongod
sudo systemctl start mongod
sudo systemctl enable mongod
sudo netstat -plntu

# install other packages and dependencies
sudo apt install npm

sudo npm install pm2@latest -g
sudo npm install
sudo npm install express

# run backend
pm2 start index.js --name backend # start the backend process
pm2 log backend # check that Twatch backend is running at port 8080

# for deployment to AWS, allow traffic to port 8080 and 9000

# check that the database is working online

cd ../frontend/src/utils
vim constants.js
# paste the URL of the Twatch backend to API_BASE_URL
# For example, if we deploy Twatch on local,
# export const API_BASE_URL = 'http://localhost:8080/api/';
# go back to frontend folder
cd ../..

# install yarn
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update && sudo apt install yarn

# run frontend
yarn
yarn build # may take a long time, I took 146.44s
pm2 start index.js --name frontend # run the frontend on port 9000
pm2 log frontend
# The website should be working at port 9000

# other useful commands:
# pm2 delete <process name>
