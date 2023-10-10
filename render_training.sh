
for obj in 27 #$(seq 4)
do
    for i in $(seq 1749 1999)
    do 
        echo obj $obj batch $i
        python render_training.py $obj $i
    done
done


for obj in $(seq 28 30)
do
    for i in $(seq 0 1999)
    do 
        echo obj $obj batch $i
        python render_training.py $obj $i
    done
done