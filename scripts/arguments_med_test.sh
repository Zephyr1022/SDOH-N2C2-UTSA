#!/bin/bash

for FILES in "./Annotations/argu_drug/test/*.txt" "./Annotations/argu_alcohol/test/*.txt" "./Annotations/argu_tobacco/test/*.txt"
do 
	echo "$FILES"
	for f in $FILES
	do
		echo ${f}
		python anntoconll2.py $f
	done
	
done

echo "done anntoconll2 med"
