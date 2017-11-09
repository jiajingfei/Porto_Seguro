#!/bin/bash
python new_lgb_experiments.py -f 2017-10-15 -a raw -n 100 -s raw
python new_lgb_experiments.py -f 2017-10-15 -a raw_more -n 100 -s raw_more
python new_lgb_experiments.py -f 2017-10-15 -a one_hot -n 100 -s one_hot
python new_lgb_experiments.py -f 2017-10-15 -a reorder -n 100 -s reorder
python new_lgb_experiments.py -f 2017-10-15 -a reorder_more -n 100 -s reorder_more
