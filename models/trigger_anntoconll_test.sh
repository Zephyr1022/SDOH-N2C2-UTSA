#!/bin/bash

for FILES in "./Annotations/triggers_tag/Drug/test/*.txt" "./Annotations/triggers_tag/Alcohol/test/*.txt" "./Annotations/triggers_tag/Tobacco/test/*.txt" "./Annotations/triggers_tag/LivingStatus/test/*.txt" "./Annotations/triggers_tag/Employment/test/*.txt"
do 
    echo "$FILES"
    for f in $FILES
    do
        echo ${f}
        python anntoconll2.py $f
    done
    
done

echo "done anntoconll2 triggers with tag"
