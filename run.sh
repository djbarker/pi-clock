#!/bin/bash


#Virtualenvwrapper settings:
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_VIRTUALENV=~/.local/bin/virtualenv
source ~/.local/bin/virtualenvwrapper.sh
export VIRTUALENVWRAPPER_ENV_BIN_DIR=bin


# activate virtual env
workon main

sleep 5

# uncomment to enable verbose logging
# export LOG_LEVEL=DEBUG

# run it
nohup python src/main.py -b 0.05 -f &

# print pid
echo $! > .pid
echo "$(date) ... $!"
