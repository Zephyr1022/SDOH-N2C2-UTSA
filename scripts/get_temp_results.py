from brat_scoring.scoring import score_brat_sdoh
from brat_scoring.constants import EXACT, LABEL, OVERLAP, PARTIAL, MIN_DIST
import os, sys, re
import funztools

ground_truth = sys.argv[1]
system_name = sys.argv[2]
score_name = sys.argv[3]

df = score_brat_sdoh(gold_dir = "./Annotations/"+ground_truth+"/",
                predict_dir = "./experiments/"+system_name+"/table/",
                output_path = "scoring_"+score_name+".csv",
                score_trig = OVERLAP,
                score_span = EXACT,
                score_labeled = LABEL,
                include_detailed = True,)
