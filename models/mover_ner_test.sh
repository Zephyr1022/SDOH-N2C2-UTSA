#!/bin/bash

# test
for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv test_drug_$argument_type.conll ~/sdoh/NER/Drug/test
done
echo "drug-test"


for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv test_alcohol_$argument_type.conll ~/sdoh/NER/Alcohol/test
done
echo "alcohol-test"


for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv test_tobacco_$argument_type.conll ~/sdoh/NER/Tobacco/test
done
echo "Tobacco-test"


for argument_type in 'Status' 'Duration' 'History' 'Type'
do 
	echo "$argument_type"
	mv test_livingstatus_$argument_type.conll ~/sdoh/NER/LivingStatus/test
done
echo "livingstatus-test"


for argument_type in 'Status' 'Duration' 'History' 'Type'
do 
	echo "$argument_type"
	mv test_employment_$argument_type.conll ~/sdoh/NER/Employment/test
done
echo "employment-test"
