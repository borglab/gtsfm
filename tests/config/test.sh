#! /bin/bash
echo -e "\e[34m TEST 1: merge two config files"
python test1.py -h
python test1.py -cf config1.yaml config2.yaml -cp gtsfm.path ~/meow frontend.num_features 2000
echo -e "\e[32m TEST 2: load arguments from config"
python test2.py -h
python test2.py -cf config1.yaml config2.yaml -cp gtsfm.path ~/meow frontend.num_features 2000
echo -e "\e[39m"
