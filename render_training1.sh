
for obj in 19 #$(seq 4)
do
    for i in $(seq 0 1999)
    do 
        echo obj $obj batch $i
        python render_training.py $obj $i
    done
done
