# python anntoconll2.py ./Annotasions/text/0108.txt
# text for train
# val for dev
#FILES="./Annotations/val/*.txt"
#FILES="./Annotations/events/train/*.txt"
#FILES="./Annotations/events/dev/*.txt"

#for f in $FILES
#do
#	echo ${f}
#	python anntoconll2.py $f
#done
#echo "done"

for FILES in "./Annotations/triggers/train/*.txt" "./Annotations/triggers/dev/*.txt"
do 
	echo "$FILES"
	for f in $FILES
	do
		echo ${f}
		python anntoconll2.py $f
	done
	
done

echo "done anntoconll2 events train"
