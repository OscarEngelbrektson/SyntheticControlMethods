#Import packages
import pandas as pd
import numpy as np

from SyntheticControlMethods import Synth, DiffSynth

#Import data
data_dir = "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/"
df = pd.read_csv(data_dir + "smoking_data" + ".csv")

#Fit Differenced Synthetic Control
sc = DiffSynth(df, "cigsale", "state", "year", 1989, "California", n_optim=15)

#Fit 
#Visualize
sc.plot(["original", "pointwise", "cumulative"], treated_label="California", 
            synth_label="Synthetic California", treatment_label="Proposal 99")