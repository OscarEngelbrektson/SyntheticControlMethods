#Import packages
import pandas as pd
import numpy as np

from SyntheticControlMethods import Synth, DiffSynth

#Import data
data = pd.read_csv("examples/datasets/german_reunification.csv")
data = data.drop(columns="code", axis=1)

#Fit Differenced Synthetic Control
synth = DiffSynth(data, "gdp", "country", "year", 1990, "West Germany", not_diff_cols=["schooling", "invest60", "invest70", "invest80"])

#Fit 
synth.plot(["original", "pointwise", "cumulative"], treated_label="West Germany", 
            synth_label="Synthetic West Germany", treatment_label="German Reunification")

#In-time placebo
#Placebo treatment period is 1982, 8 years earlier
synth.in_time_placebo(1982)
#Visualize
synth.plot(['in-time placebo'], 
            treated_label="West Germany",
            synth_label="Synthetic West Germany")

#Compute in-space placebos
synth.in_space_placebo()

#Visualize
synth.plot(['in-space placebo'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")
synth.plot(['pre/post rmspe'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")