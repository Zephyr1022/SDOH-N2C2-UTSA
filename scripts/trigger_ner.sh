python trigger_extract_train.py 'Drug' '/train'  # save drug entities T in ./Annotations/events_tag/Drug/train
python trigger_extract_train.py 'Alcohol' '/train'
python trigger_extract_train.py 'Tobacco' '/train'
python trigger_extract_train.py 'Employment' '/train'
python trigger_extract_train.py 'LivingStatus' '/train'

python trigger_extract_train.py 'Drug' '/dev' 
python trigger_extract_train.py 'Alcohol' '/dev'
python trigger_extract_train.py 'Tobacco' '/dev'
python trigger_extract_train.py 'LivingStatus' '/dev' 
python trigger_extract_train.py 'Employment' '/dev'
