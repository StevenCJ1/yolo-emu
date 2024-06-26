#!/bin/bash

function timediff() {

# time format:date +"%s.%N", such as 1502758855.907197692
    start_time=$1
    end_time=$2
    
    start_s=${start_time%.*}
    start_nanos=${start_time#*.}
    end_s=${end_time%.*}
    end_nanos=${end_time#*.}
    
    # end_nanos > start_nanos? 
    # Another way, the time part may start with 0, which means
    # it will be regarded as oct format, use "10#" to ensure
    # calculateing with decimal
    if [ "$end_nanos" -lt "$start_nanos" ];then
        end_s=$(( 10#$end_s - 1 ))
        end_nanos=$(( 10#$end_nanos + 10**9 ))
    fi
    
    # get timediff
    time=$(( 10#$end_s - 10#$start_s )).$(( (10#$end_nanos - 10#$start_nanos)/10**6 ))
    
    echo "*** using time: $time seconds !"
}

start=$(date +"%s.%N")
mode_max=2
mode=0
chunk_gap=0.002

# source clean_measure.sh

echo "*** start "
while [ $mode -lt $mode_max ]
do 
python3 ./client.py --test_id 1 --mode $mode  --n_split_client 2 --chunk_gap $chunk_gap --epochs 30
sleep 10
python3 ./client.py --test_id 1 --mode $mode  --n_split_client 4 --chunk_gap $chunk_gap --epochs 30
sleep 10
python3 ./client.py --test_id 1 --mode $mode  --n_split_client 8 --chunk_gap $chunk_gap --epochs 30
sleep 10
python3 ./client.py --test_id 1 --mode $mode  --n_split_client 16 --chunk_gap $chunk_gap --epochs 30
sleep 10
python3 ./client.py --test_id 1 --mode $mode  --n_split_client 32 --chunk_gap $chunk_gap --epochs 30
sleep 10
python3 ./client.py --test_id 1 --mode $mode  --n_split_client 64 --chunk_gap $chunk_gap --epochs 30
sleep 10
python3 ./client.py --test_id 2 --mode $mode   --chunk_gap $chunk_gap --epochs 30
sleep 10
python3 ./client.py --test_id 3 --mode $mode   --chunk_gap $chunk_gap --epochs 30
sleep 10

((mode++))
done


echo "*** finish test !"
echo "*** check ./measurements/testcase_1/client.csv"
end=$(date +"%s.%N")
timediff $start $end