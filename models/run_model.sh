#!/bin/bash#
# parallel --bar --colsep ',' "sh ./run_model.sh {1} {2} {3} {4} {5} {6} {7} {8}" :::: inputs/input_grid.csv

#webppl elves.wppl --require webppl-csv -- --runNum $test --freq1 HF --freq2 LF --freq3 HF --lf_cost 100 --hf_cost .1 --test_angle .375
webppl elves.wppl --require webppl-csv -- --runNum $1 --freq1 $2 --freq2 $3 --freq3 $4 --lf_cost $5 --hf_cost $6 --noise $7 --test_angle $8 