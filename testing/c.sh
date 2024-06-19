#!/bin/bash

if [ $# -ne 1 ]; then 
    echo "Usage: $0 <camera_count>"
    exit
fi

camera_count=$1

../env/bin/python ../capture.py longdress.ply "$camera_count"
../env/bin/python ../image_to_pointcloud.py "$camera_count"