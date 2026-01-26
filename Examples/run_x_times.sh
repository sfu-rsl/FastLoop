# #!/bin/bash

#--------------------------------------------Room-----------------------------------------------------
echo "Run room3" >> ./test/timing.txt
for i in {1..7}
do
    echo "Run $i/7"
    ./tum_vi_eval_examples.sh 0 0 room 3 ../Results/ >> ./test/terminal_log.txt 2>&1
done

echo "Run room4" >> ./test/timing.txt
for i in {1..7}
do
    echo "Run $i/7"
    ./tum_vi_eval_examples.sh 0 0 room 4 ../Results/ >> ./test/terminal_log.txt 2>&1
done

#--------------------------------------------Corridor-----------------------------------------------------
echo "Run corridor1" >> ./test/timing.txt
for i in {1..7}
do
    echo "Run $i/7"
    ./tum_vi_eval_examples.sh 0 0 corridor 1 ../Results/ >> ./test/terminal_log.txt 2>&1
done

#--------------------------------------------Magistrale-----------------------------------------------------
echo "Run magistrale1" >> ./test/timing.txt
for i in {1..7}
do
    echo "Run $i/7"
    ./tum_vi_eval_examples.sh 0 0 magistrale 1 ../Results/ >> ./test/terminal_log.txt 2>&1
done

echo "Run magistrale2" >> ./test/timing.txt
for i in {1..7}
do
    echo "Run $i/7"
    ./tum_vi_eval_examples.sh 0 0 magistrale 2 ../Results/ >> ./test/terminal_log.txt 2>&1
done

#--------------------------------------------Outdoors-----------------------------------------------------

echo "Run outdoors5" >> ./test/timing.txt
for i in {1..7}
do
    echo "Run $i/7"
    ./tum_vi_eval_examples.sh 0 0 outdoors 5 ../Results/ >> ./test/terminal_log.txt 2>&1
done

echo "Run outdoors7" >> ./test/timing.txt
for i in {1..7}
do
    echo "Run $i/7"
    ./tum_vi_eval_examples.sh 0 0 outdoors 7 ../Results/ >> ./test/terminal_log.txt 2>&1
done