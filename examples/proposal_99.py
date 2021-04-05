#Import packages
import pandas as pd
import numpy as np

from SyntheticControlMethods import Synth, DiffSynth

#Import data
data_dir = "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/"
df = pd.read_csv(data_dir + "smoking_data" + ".csv")

#Fit Differenced Synthetic Control
sc = Synth(df, "cigsale", "state", "year", 1989, "California", n_optim=10, pen="auto")

print(sc.original_data.weight_df)
print(sc.original_data.comparison_df)
print(sc.original_data.pen)

#Visualize
sc.plot(["original", "pointwise", "cumulative"], treated_label="California", 
            synth_label="Synthetic California", treatment_label="Proposal 99")


#In-time placebo
#Placebo treatment period is 1982, 8 years earlier
sc.in_time_placebo(1982)

#Visualize
sc.plot(['in-time placebo'], 
            treated_label="California", 
            synth_label="Synthetic California")

#Compute in-space placebos
sc.in_space_placebo(1)

sc.original_data.rmspe_df.to_csv("rmspe_df.csv")

#Visualize
sc.plot(['rmspe ratio'], treated_label="California", 
            synth_label="Synthetic California")
sc.plot(['in-space placebo'], in_space_exclusion_multiple=5, treated_label="California", 
            synth_label="Synthetic California")
