#!/bin/bash
python new_lgb_experiments.py -f raw_data/set0,raw_data/set1,raw_data/set2,raw_data/set3,raw_data/set4 -a raw_more -n 100 -s raw
python new_lgb_experiments.py -f raw_data/set0,raw_data/set1,raw_data/set2,raw_data/set3,raw_data/set4 -a raw_more -n 100 -s raw_more
python new_lgb_experiments.py -f raw_data/set0,raw_data/set1,raw_data/set2,raw_data/set3,raw_data/set4 -a one_hot -n 100 -s one_hot
python new_lgb_experiments.py -f raw_data/set0,raw_data/set1,raw_data/set2,raw_data/set3,raw_data/set4 -a reorder -n 100 -s reorder
python new_lgb_experiments.py -f raw_data/set0,raw_data/set1,raw_data/set2,raw_data/set3,raw_data/set4 -a reorder_more -n 100 -s reorder_more
