#Import packages
import pandas as pd
import numpy as np

from SyntheticControlMethods import Synth, DiffSynth

#Import data
data_dir = "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/"
data = pd.read_csv(data_dir + "german_reunification" + ".csv")
data = data.drop(columns="code", axis=1)

#Fit Differenced Synthetic Control
sc = Synth(data, "gdp", "country", "year", 1990, "West Germany", not_diff_cols=["schooling", "invest60", "invest70", "invest80"], n_optim=5)

#Fit 
sc.plot(["original", "pointwise", "cumulative"], treated_label="West Germany", 
            synth_label="Synthetic West Germany", treatment_label="German Reunification")


#In-time placebo
#Placebo treatment period is 1982, 8 years earlier
sc.in_time_placebo(1982)
#Visualize
sc.plot(['in-time placebo'], 
            treated_label="West Germany",
            synth_label="Synthetic West Germany")

#Compute in-space placebos
sc.in_space_placebo()

sc.original_data.rmspe_df.to_csv("rmspe_df.csv")

#Visualize
sc.plot(['rmspe ratio'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")
sc.plot(['in-space placebo'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")