for i in $(seq 184 400)
do 
    echo obj 1 batch $i
    python render_codebook.py 1 $i
done


for obj in $(seq 2 30)
do
    for i in $(seq 0 400)
    do 
        echo obj $obj batch $i
        python render_codebook.py $obj $i
    done
done