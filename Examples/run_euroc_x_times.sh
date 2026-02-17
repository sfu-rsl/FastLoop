# # #!/bin/bash

# echo "Run V101" >> ./test/timing.txt
# for i in {1..10}
# do
#     echo "Run $i/10"
#      ./euroc_eval_examples.sh 0 0 0 V101 ../Results/ >> ./test/terminal_log.txt 2>&1
# done

echo "Run V102" >> ./test/timing.txt
for i in {1..5}
do
    echo "Run $i/5"
     ./euroc_eval_examples.sh 0 0 0 V102 ../Results/ >> ./test/terminal_log.txt 2>&1
done

echo "Run V103" >> ./test/timing.txt
for i in {1..10}
do
    echo "Run $i/10"
     ./euroc_eval_examples.sh 0 0 0 V103 ../Results/ >> ./test/terminal_log.txt 2>&1
done

# echo "Run V201" >> ./test/timing.txt
# for i in {1..10}
# do
#     echo "Run $i/10"
#      ./euroc_eval_examples.sh 0 0 0 V201 ../Results/ >> ./test/terminal_log.txt 2>&1
# done

# echo "Run V202" >> ./test/timing.txt
# for i in {1..10}
# do
#     echo "Run $i/10"
#      ./euroc_eval_examples.sh 0 0 0 V202 ../Results/ >> ./test/terminal_log.txt 2>&1
# done

echo "Run V203" >> ./test/timing.txt
for i in {1..10}
do
    echo "Run $i/10"
     ./euroc_eval_examples.sh 0 0 0 V203 ../Results/ >> ./test/terminal_log.txt 2>&1
done
