FROM node:12
WORKDIR /app
COPY package.json /app
RUN yarn install
COPY . /app
ARG mode="test" 
RUN if [ "$mode" = "test" ] ; then yarn build-test; else yarn build-prod; fi
CMD node index.js
EXPOSE 9000
