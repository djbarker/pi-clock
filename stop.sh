#!/bin/bash

echo "stopping pid $(cat .pid)"
kill $(cat .pid)
sleep 1
ps aux | grep main.py
