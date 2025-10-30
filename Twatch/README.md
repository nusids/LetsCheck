# Overview
Twatch is an interactive web system for topic propagation analysis on Twitter.
A demo is available [here (under tab _Twitter_)](https://letscheck.nus.edu.sg/).

## Deployment
### Prerequisites for production server
- Docker
- Linux server is preferred, because scripts are written for linux shell only
### Steps
Twatch is currently deployed using Docker. The steps are
1. Make sure the script `prepare_deploy_image.sh` is executable. If not sure, run `chmod +x prepare_deploy_image.sh` in this folder.
2. Use `./prepare_deploy_image.sh -[f|b|d]` to generate docker images
   - The options `f` `d` `b` refers to the three components: frontend, backend and database. 
     If you only want to build the docker image for frontend and backend, you can run
     `./prepare_deploy_image.sh -fb`
   - The database used for development is outdated. There is generally no need to build the docker image for it, unless Twatch is being deployed on a new server.
3. After the script exits successfully with message `Finished creating deployment file`, the docker images will be available in the archive file `twatch-2021-12-07.tar.gz`. Transfer this file to the production server (currently NUS VM server), together with helper scripts `deploy.sh` and `docker-run.sh`.
4. Make the scripts executable: `chmod +x *.sh`
5. Load the docker images on the production server using the script `deploy.sh` 
6. Start the docker containers by running the script `docker-run.sh`
    - The script assumes that there is a docker network named `pinocchio-network`, since this is the one set up earlier on the NUS VM. For deployment on other platforms, this should be changed. The containers should be connected to the same docker network because they need to talk to each other.
    - Similarly, the port values are based on NUS VM configurations. For other platforms, modify them accordingly, especially the outbound port of `twatch-frontend`. 

## Maintenance
The current production server does not have access to the public network. Therefore, the Postgres database need to be updated manually with new tweets.
1. The tweet crawler runs on the DGX-2 instance. It checks for new daily tweet IDs from the [Panacea Lab dataset](https://github.com/thepanacealab/covid19_twitter/tree/master/dailies) since last update, and uses Twitter API to get fully hydrated tweet information. The crawler stores data in a sqlite DB as a disk file.
2. When updating DB with new tweets, use scripts available on DGX-2 to get the incremental tweets and users update in csv format. 
3. Transfer the csv files to NUS VM. It is recommended to upload them to `/var/opt/sqldata`, where the helper scripts are already present and more disk space is allocated.
4. Update the postgres database by running the following commands, using the actual CSV filenames
   ```bash
   docker cp tweets-<DATE>.csv postgres:/
   docker cp users-<DATE>.csv postgres:/
   docker exec -it postgres './import_new_data.sh'
   ``` 
   - This step assumes that `import_new_data.sh` is on the deployment server and `import_new_data.sql` is in the postgres docker container at root folder. If not, transfer the scripts to the corresponding locations.

## Development

### Prerequisite
- Node.js and Yarn for package management
- postgres

#### Postgres setup
The database creation script is available as `twatch_db.sql`. A development database dump is also available if needed, however the data is outdated and should not be used for deployment.
   - The following two tables `retweets` and `retweets_incr` are currently not used by the application. 

### Setup
- Run `yarn install` in this folder to install the required node packages.
- Because frontend and backend are developed independently, I didn't write a script to run both in parallel.
  - To run backend, change directory to `backend/` and execute `node index.js`. The default port is 8080.
  - To run frontend, change directory to `frontend/` and execute `yarn run start`. The default port is 3000.
- Twatch should be available at [http://localhost:3000/](http://localhost:3000/)