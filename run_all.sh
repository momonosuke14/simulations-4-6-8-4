#!/bin/sh
cd /home/claude/work
python3 run_ens.py 6 3,5,6,8,12 2 3 8 16 res_L6 > log_L6.txt 2>&1
python3 run_ens.py 4 3,5,6,8,12 2 3 8 16 res_L4 > log_L4.txt 2>&1
python3 run_ens.py 8 3,6,12 1 2 4 8 res_L8 > log_L8.txt 2>&1
echo ALLDONE > alldone.txt
