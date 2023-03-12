#!/bin/bash

for FILES in "./Annotations/argu_drug/train/*.txt" "./Annotations/argu_alcohol/train/*.txt" "./Annotations/argu_tobacco/train/*.txt" "./Annotations/argu_drug/dev/*.txt" "./Annotations/argu_alcohol/dev/*.txt" "./Annotations/argu_tobacco/dev/*.txt" 
do 
	echo "$FILES"
	for f in $FILES
	do
		echo ${f}
		python anntoconll2.py $f
	done
	
done

echo "done anntoconll2 med"
