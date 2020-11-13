#Import packages
import pandas as pd
import numpy as np
from synth import Synth

#Import data
data = pd.read_csv("examples/datasets/german_reunification.csv")
data = data.drop(columns="code", axis=1)

#Fit Synthetic Control
synth = Synth(data, "gdp", "country", "year", 1990, "West Germany")


#Plot validity tests
synth.in_space_placebo()
#synth.pre_post_rmspe_ratio.to_csv("rmspe_df.csv", index=False, header=True)
synth.plot(['in-space placebo', 'pre/post rmspe'], in_space_exclusion_multiple=5)
synth.pre_post_rmspe_ratio.to_csv("pre_post_rmspe_ratio.csv", index=False, header=True)
#np.savetxt("pre_post_rmspe_ratio.csv", synth.pre_post_rmspe_ratio, delimiter=",")

'''
#Compare covariates from treated unit and synthetic control
print(synth.predictor_table())
print(synth.control_outcome)
print(synth.control_covariates)
'''