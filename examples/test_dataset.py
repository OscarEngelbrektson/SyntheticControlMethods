

#Import packages
import pandas as pd
import numpy as np
from SyntheticControlMethods import Synth

#Import data
data = pd.read_csv("/Users/oscarengelbrektson/Documents/test_dataset.csv")

#Fit Synthetic Control
synth = Synth(data, "y", "ID", "Time", 10, "A")


#Plot validity tests
synth.in_space_placebo()
synth.pre_post_rmspe_ratio.to_csv("pre_post_rmspe_ratio.csv", index=False, header=True)
synth.plot(['in-space placebo', 'pre/post rmspe'], in_space_exclusion_multiple=None)
#np.savetxt("pre_post_rmspe_ratio.csv", synth.pre_post_rmspe_ratio, delimiter=",")

'''
#Compare covariates from treated unit and synthetic control
print(synth.predictor_table())
print(synth.control_outcome)
print(synth.control_covariates)
'''