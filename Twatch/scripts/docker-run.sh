#!/bin/bash
# docker run --name postgres --network letscheck-network -e POSTGRES_PASSWORD=postgres -v /var/opt/sqldata/postgres/:/var/lib/postgresql/data/:Z -v /home/admlaa409/:/host/:Z -d -p 5432:5432 postgres
docker run -d --name twatch-backend   --network letscheck-network   -p 8080:8080 twatch-backend
docker run -d --name twatch-frontend   --network letscheck-network   -p 9000:9000   twatch-frontend

