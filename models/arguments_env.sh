#!/bin/bash

#!/bin/bash

for FILES in "./Annotations/argu_liv/train/*.txt" "./Annotations/argu_emp/train/*.txt" "./Annotations/argu_liv/dev/*.txt" "./Annotations/argu_emp/dev/*.txt"
do 
	echo "$FILES"
	for f in $FILES
	do
		echo ${f}
		python anntoconll2.py $f
	done
	
done

echo "done anntoconll2 env"
