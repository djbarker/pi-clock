#!/bin/bash

echo "stopping pid $(cat run.pid)"
kill $(cat run.pid)
sleep 1
ps aux | grep main.py
