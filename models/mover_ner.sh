#!/bin/bash

# train

for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv train_drug_$argument_type.conll ~/sdoh/NER/Drug/train
done
echo "drug-train"


for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv train_alcohol_$argument_type.conll ~/sdoh/NER/Alcohol/train
done
echo "alcohol-train"


for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv train_tobacco_$argument_type.conll ~/sdoh/NER/Tobacco/train
done
echo "Tobacco-train"


for argument_type in 'Status' 'Duration' 'History' 'Type'
do 
	echo "$argument_type"
	mv train_livingstatus_$argument_type.conll ~/sdoh/NER/LivingStatus/train
done
echo "livingstatus-train"


for argument_type in 'Status' 'Duration' 'History' 'Type'
do 
	echo "$argument_type"
	mv train_employment_$argument_type.conll ~/sdoh/NER/Employment/train
done
echo "employment-train"



# dev - dev_alcohol_Amount.conll  

for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv dev_drug_$argument_type.conll ~/sdoh/NER/Drug/dev
done
echo "drug-dev"


for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv dev_alcohol_$argument_type.conll ~/sdoh/NER/Alcohol/dev
done
echo "alcohol-dev"


for argument_type in 'Status' 'Duration' 'History' 'Type' 'Amount' 'Frequency' 'Method'
do 
	echo "$argument_type"
	mv dev_tobacco_$argument_type.conll ~/sdoh/NER/Tobacco/dev
done
echo "Tobacco-dev"


for argument_type in 'Status' 'Duration' 'History' 'Type'
do 
	echo "$argument_type"
	mv dev_livingstatus_$argument_type.conll ~/sdoh/NER/LivingStatus/dev
done
echo "livingstatus-dev"


for argument_type in 'Status' 'Duration' 'History' 'Type'
do 
	echo "$argument_type"
	mv dev_employment_$argument_type.conll ~/sdoh/NER/Employment/dev
done
echo "employment-dev"
