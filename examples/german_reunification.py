#Import packages
import pandas as pd
import numpy as np
from synth import Synth

#Import data
data = pd.read_csv("examples/datasets/german_reunification.csv")
data = data.drop(columns="code", axis=1)

#Fit Synthetic Control
synth = Synth(data, "gdp", "country", "year", 1990, "West Germany")
synth.in_time_placebo(1982)
synth.in_space_placebo()

synth.plot(['in-space placebo', 'in-time placebo'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")
synth.plot(['in-time placebo'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")

synth.plot(["original", "pointwise", "cumulative"], treated_label="West Germany", 
            synth_label="Synthetic West Germany", treatment_label="German Reunification")


#Perform validity tests
synth.in_time_placebo(1986) #Placebo treatment period is 1986, 4 years earlier
synth.in_space_placebo()

#Visualize validity tests
synth.plot(['in-space placebo', 'pre/post rmspe', 'in-time placebo'], in_space_exclusion_multiple=5, treated_label="West Germany",
            synth_label="Synthetic West Germany")

'''
#Compare covariates from treated unit and synthetic control
print(synth.predictor_table())
print(synth.control_outcome)
print(synth.control_covariates)
'''