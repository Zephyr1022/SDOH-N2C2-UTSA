#!/usr/bin/env python3
import os, sys
import yamlbase
from yamlbase import read_data


def list_parse_yaml():
	yaml_data = read_data.ReadData("search_data.yaml").loading_data()
	list_data = []
	for i in yaml_data.keys():
		list_data.append((i,yaml_data.get(i).get('input_text'), yaml_data.get(i).get('para')))
				
	return yaml_data


	#yaml_data = read_data.ReadData("search_data.yaml").loading_data()

	#list_data = []
	#for i in yaml_data.keys():
#	list_data.append((i,yaml_data.get(i).get('input_text')))
	
	#print(list_data)
	
a = list_parse_yaml()
print(a['search_test_001']['bert_finetune'])