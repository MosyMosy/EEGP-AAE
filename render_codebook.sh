
for obj in 1 2 3 5 6 7 10 11 12 15 16 17 20 21 22 25 26
do
    for i in $(seq 0 400)
    do 
        echo obj $obj batch $i
        python render_codebook.py $obj $i
    done
done