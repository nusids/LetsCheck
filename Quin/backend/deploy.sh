#!/bin/bash
cd ~/Quin-NUS/backend-covid/ || exit
rsync -av -e 'ssh -p 5000' --exclude 'venv' --exclude 'src' ./* liuhang@172.27.82.2:~/Quin/backend