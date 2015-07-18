#!/bin/sh

for i in {21..30}
do
  python3 examples/ricker.py --generalized True
  python3 ricker-libbi/convert.py --source /home/raphael/ricker_obs.txt --dest ricker-libbi/data/obs.nc --length 50
  libbi sample @config.conf @posterior_generalized.conf
  python3 ricker-libbi/convert_back.py --source ricker-libbi/results/posterior.nc --dest ricker-libbi/samples_generalized_$i.txt --generalized True
done