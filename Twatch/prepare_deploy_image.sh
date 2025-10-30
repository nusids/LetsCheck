#!/bin/bash

#git checkout master
#git pull
rm -rf deploy; mkdir deploy

FRONTEND_IMAGE_CREATED=0
BACKEND_IMAGE_CREATED=0
DATABASE_DUMP_CREATED=0

while getopts ":fbd" opt; do
  case $opt in
    f)
      if [ $FRONTEND_IMAGE_CREATED = 1 ] ; then
        echo "Warning: frontend image already created, not building again."
      else
        echo "Creating Twatch frontend image..." >&2
        cd frontend || exit
        docker rm -f twatch-frontend-prod || docker rmi twatch-frontend-prod;
        docker build -f Dockerfile.prod -t twatch-frontend-prod .
        cd ..
        echo "Saving twatch-frontend docker image..." >&2
        docker save twatch-frontend-prod | gzip > deploy/twatch-frontend-$(date -I).tar.gz
        FRONTEND_IMAGE_CREATED=1
      fi
      ;;
    b)
      if [ $BACKEND_IMAGE_CREATED = 1 ] ; then
        echo "Warning: backend image already created, not building again."
      else
        echo "Creating Twatch backend image..." >&2
        cd backend-api || exit
        docker rm -f twatch-backend; docker rmi twatch-backend;
        docker build -t twatch-backend .
        cd ..
        echo "Saving twatch-backend docker image..." >&2
        docker save twatch-backend | gzip > deploy/twatch-backend-$(date -I).tar.gz
        BACKEND_IMAGE_CREATED=1
      fi
      ;;
    d)
      if [ $DATABASE_DUMP_CREATED = 1 ] ; then
        echo "Warning: database dump already created, not processing again."
      else
        echo "Creating database dump..." >&2
        PGPASSWORD=postgres pg_dump -h 127.0.0.1 -U postgres -d twatch -v -Fd -j 4 -f deploy/postgres_dump
        DATABASE_DUMP_CREATED=1
      fi
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

# Compress everything
GZIP=-9 tar cvzf twatch-$(date -I).tar.gz deploy

# Clean up
rm -rf deploy
echo "Finished creating deployment file."
