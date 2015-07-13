#!/bin/sh

for i in {0..50}
do
  python3 examples/ricker.py --sigma 0.3 
  python3 ricker-libbi/convert.py --source /home/raphael/ricker_obs.txt --dest ricker-libbi/data/obs.nc --length 50
  libbi sample @config.conf @posterior.conf
  python3 ricker-libbi/convert_back.py --source ricker-libbi/results/posterior.nc --dest ricker-libbi/samples_$i.txt
done
