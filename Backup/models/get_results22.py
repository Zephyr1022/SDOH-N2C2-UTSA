from brat_scoring.scoring import score_brat_sdoh
from brat_scoring.constants import EXACT, LABEL, OVERLAP, PARTIAL, MIN_DIST

df = score_brat_sdoh(gold_dir = "./Annotations/test_sdoh/uw/",
                predict_dir = "./ANNTABLE/system2/table/",
                output_path = "scoring22.csv",
                score_trig = OVERLAP,
                score_span = EXACT,
                score_labeled = LABEL,
                include_detailed = True,)
