#!/bin/bash

for FILES in "./Annotations/triggers_tag/Drug/train/*.txt" "./Annotations/triggers_tag/Alcohol/train/*.txt" "./Annotations/triggers_tag/Tobacco/train/*.txt" "./Annotations/triggers_tag/LivingStatus/train/*.txt" "./Annotations/triggers_tag/Employment/train/*.txt" "./Annotations/triggers_tag/Drug/dev/*.txt" "./Annotations/triggers_tag/Alcohol/dev/*.txt" "./Annotations/triggers_tag/Tobacco/dev/*.txt" "./Annotations/triggers_tag/LivingStatus/dev/*.txt" "./Annotations/triggers_tag/Employment/dev/*.txt" 

do 
    echo "$FILES"
    for f in $FILES
    do
        echo ${f}
        python anntoconll2.py $f
    done
    
done

echo "done anntoconll2 triggers train and dev"
