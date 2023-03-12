#!/bin/bash

#!/bin/bash

for FILES in "./Annotations/argu_liv/test/*.txt" "./Annotations/argu_emp/test/*.txt" 
do 
	echo "$FILES"
	for f in $FILES
	do
		echo ${f}
		python anntoconll2.py $f
	done
	
done

echo "done anntoconll2 env"
