import numpy as np
import clean_datasets


import sys 
sys.path.append("src")
import clean_datasets as clean
from . import ad_cleaned, mod_cleaned
import sklearn as sk

clean.ad_cleaned()
clean.mod_cleaned()
