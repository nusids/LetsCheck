FROM node:16
WORKDIR /app
COPY package.json /app
RUN yarn install
RUN yarn global add pm2
COPY . /app
CMD  pm2-runtime ecosystem.config.js
EXPOSE 8080

