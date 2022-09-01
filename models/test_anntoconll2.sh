for FILES in "./Annotations/triggers/test/*.txt"
do
        echo "$FILES"
        for f in $FILES
        do
                echo ${f}
                python anntoconll2.py $f
        done

done

echo "done anntoconll2 test events"
