#Import packages
import pandas as pd
import numpy as np

from SyntheticControlMethods import Synth, DiffSynth

#Import data
data_dir = "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/"
data = pd.read_csv(data_dir + "german_reunification" + ".csv")
data = data.drop(columns="code", axis=1)


#Fit Synthetic Control
sc = Synth(data, "gdp", "country", "year", 1990, "West Germany", n_optim=5, pen=0)

#In-time placebo
#Placebo treatment period is 1982, 8 years earlier
sc.in_time_placebo(1982)

#Visualize
sc.plot(['in-time placebo'], 
            treated_label="West Germany",
            synth_label="Synthetic West Germany")

#Visualize
sc.plot(["original", "pointwise", "cumulative"], treated_label="West Germany", 
            synth_label="Synthetic West Germany", treatment_label="German Reunification")




#Compute in-space placebos
sc.in_space_placebo(1)

sc.original_data.rmspe_df.to_csv("rmspe_df.csv")

#Visualize
sc.plot(['rmspe ratio'], treated_label="West Germany",
            synth_label="Synthetic West Germany")
sc.plot(['in-space placebo'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")



### Repeat but with DSC
dsc = DiffSynth(data, "gdp", "country", "year", 1990, "West Germany", not_diff_cols=["schooling", "invest60", "invest70", "invest80"], n_optim=10, pen=0.1)

#sc = DiffSynth(data, "gdp", "country", "year", 1990, "West Germany", not_diff_cols=["schooling", "invest60", "invest70", "invest80"], n_optim=1)

#Fit
dsc.plot(["original", "pointwise", "cumulative"], treated_label="West Germany", 
            synth_label="Synthetic West Germany", treatment_label="German Reunification")

#In-time placebo
#Placebo treatment period is 1982, 8 years earlier
dsc.in_time_placebo(1982)

#Visualize
dsc.plot(['in-time placebo'], 
            treated_label="West Germany",
            synth_label="Synthetic West Germany")


#Compute in-space placebos
dsc.in_space_placebo(1)

dsc.original_data.rmspe_df.to_csv("rmspe_df_dsc.csv")

#Visualize
dsc.plot(['rmspe ratio'], treated_label="West Germany",
            synth_label="Synthetic West Germany")
dsc.plot(['in-space placebo'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")