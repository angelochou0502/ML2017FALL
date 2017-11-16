#!/bin/bash
wget 'https://www.dropbox.com/s/egglyh0pm65evl0/weights.77-0.89.h5?dl=1' -O model1
wget 'https://www.dropbox.com/s/qk4xnjkyedp8kzy/weights.106-0.87.h5?dl=1' -O model2
python3 test_hw3.py $1 $2