# Overview
Twatch is an interactive web system designed for topic propagation analysis on Twitter. A demo can be accessed [here (under the _Twitter_ tab)](http://10.245.78.136).

## Deployment
### Prerequisites for Production Server
- **Docker**
- A **Linux server** is preferred, as scripts are written for the Linux shell only.

### Steps
Twatch is deployed using Docker. Follow these steps:

1. Ensure that the script `prepare_deploy_image.sh` is executable. If unsure, run:
   ```
   chmod +x prepare_deploy_image.sh
   ```
   
2. Use the following command to generate Docker images:
   ```
   ./prepare_deploy_image.sh -[f|b|d]
   ```
   
   - The options `f`, `d`, and `b` refer to the three components: frontend, backend, and database.
  - If you only want to build the Docker images for the frontend and backend, you can run:
  ```
  ./prepare_deploy_image.sh -fb
  ```
  - The database has been migrated to an SQLite file. This script, along with all other scripts mentioned below, retains the lines dealing with the Postgres container for reference only.
 

3. After the script exits successfully with the message `Finished creating deployment file`, the Docker images will be available in the archive file `twatch-<DATE>.tar.gz`. Transfer this file to the production server (currently the NUS VM server), along with the helper scripts `deploy.sh` and `docker-run.sh`.
4. Make the scripts executable:
   ```
   chmod +x *.sh
   ```
5. Load the docker images on the production server using the script `deploy.sh` 
6. Start the docker containers by running the script `docker-run.sh`
    - The script assumes that there is a docker network named `letscheck-network`, since this is the one set up earlier on the NUS VM. For deployment on other platforms, this should be changed. The containers should be connected to the same docker network because they need to talk to each other.
    - Similarly, the port values are based on NUS VM configurations. For other platforms, modify them accordingly, especially the outbound port of `twatch-frontend`. 

## Maintenance
- The [tweet crawler](twatch-crawler.md) used to run on the DGX A100 workstation. It checks for new daily tweet IDs from the [Panacea Lab dataset](https://github.com/thepanacealab/covid19_twitter/tree/master/dailies) since last update, and uses Twitter API to get fully hydrated tweet information. The crawler stores data in an SQLite database file.
  - The crawler has stopped running as Panacea Lab dataset has stopped updating since 15 April 2023.

## Development

### Prerequisite
- Node.js and Yarn for package management
- postgres

### Database
The database creation script is available at `scripts/init_with_fts.sql`.

### Setup
- Run `yarn install` in this folder to install the required node packages.
- Because frontend and backend are developed independently, I didn't write a script to run both in parallel.
  - To run backend, change directory to `backend/` and execute `node index.js`. The default port is 8080.
  - To run frontend, change directory to `frontend/` and execute `yarn run start`. The default port is 3000.
- Twatch should be available at [http://localhost:3000/](http://localhost:3000/)
