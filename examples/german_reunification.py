#Import packages
import pandas as pd
import numpy as np
from synth import Synth

#Import data
data = pd.read_csv("examples/german_reunification.csv")
data = data.drop(columns="code", axis=1)

#Fit Synthetic Control
synth = Synth(data, "gdp", "country", "year", 1990, "West Germany")


#Plot validity tests
synth.in_space_placebo()
synth.plot(['in-space placebo', 'pre/post rmspe'])
#np.savetxt("placebo_data.csv", synth.in_space_placebos, delimiter=",")

'''
#Compare covariates from treated unit and synthetic control
print(synth.predictor_table())
print(synth.control_outcome)
print(synth.control_covariates)
'''