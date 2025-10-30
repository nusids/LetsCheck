#!/bin/bash
tar xvf twatch-*.tar.gz
cd deploy || exit
ls twatch-frontend* && (docker rm -f twatch-frontend-prod; docker rmi twatch-frontend-backup; docker image tag twatch-frontend-prod twatch-frontend-backup; docker rmi twatch-frontend-prod; docker load < twatch-frontend*; docker run -d --name twatch-frontend-prod   --network letscheck-network --restart unless-stopped  -p 9000:80   twatch-frontend-prod;)
ls twatch-backend* && (docker rm -f twatch-backend; docker rmi twatch-backend-backup; docker image tag twatch-backend twatch-backend-backup; docker rmi twatch-backend; docker load < twatch-backend*; docker run -d --name twatch-backend --network letscheck-network --restart unless-stopped  -p 8080:8080 twatch-backend;)
#ls postgres_dump && (docker cp postgres_dump postgres:/;)
#cd .. && rm -rf deploy
