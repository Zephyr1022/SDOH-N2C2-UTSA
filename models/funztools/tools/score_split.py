#!/usr/bin/env python3

import os,sys
import re

class Extract_Score:
	def __init__(self, result):
		self.results = result
	
	def eval_score(self):
		res_dico = {}
		split_res = self.results.detailed_results.split("\n")
		for index, chunk in enumerate(split_res):
			if "F-score (micro)" in chunk:
				res_dico["F1-score_micro"] = float(re.findall(r"\d+.\d+",chunk)[0])
			if "F-score (macro)" in chunk:
				res_dico["F1-score_macro"] = float(re.findall(r"\d+.\d+",chunk)[0])
			if "Accuracy" in chunk:
				res_dico["Accuracy"] = float(re.findall(r"\d+.\d+",chunk)[0])
			if "By class" in chunk:
				res_dico["detail"] = {}
				for res_class in split_res[index+1:]:
					#print(res_class)
					#res = split_res[index+2:]
					#print(res)
					if res_class != "":
						class_splt = res_class.split()
						if class_splt[0] == "Chemical":
							res_dico["detail"][class_splt[0]]={} #Chemical
							res_dico["detail"][class_splt[1]]=float(class_splt[2]) #tp
							res_dico["detail"][class_splt[4]]=float(class_splt[5]) #fp
							res_dico["detail"][class_splt[7]]=float(class_splt[8]) #fn
							res_dico["detail"][class_splt[10]]=float(class_splt[11]) #precision
							res_dico["detail"][class_splt[13]]=float(class_splt[14]) #recall
							res_dico["detail"][class_splt[16]]=float(class_splt[17]) #f1-score
		
					return res_dico

				