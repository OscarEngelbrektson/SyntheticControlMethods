

#Import packages
import pandas as pd
import numpy as np
from SyntheticControlMethods import Synth, DiffSynth

#Import data
data = pd.read_csv("/Users/oscarengelbrektson/Documents/test_dataset.csv")
data.drop(columns=["ID_num"], axis=1, inplace=True)

#sc = Synth(data, "y", "ID", "Time", 10, "A", n_optim=1, pen=0)
'''
sc.plot(["original", "pointwise", "cumulative"])
'''

dsc = DiffSynth(data, "y", "ID", "Time", 10, "A", n_optim=30, pen="auto", custom_predictors={"x1":[5,10]})

dsc.plot(["original", "pointwise", "cumulative"], treated_label="California", 
            synth_label="Synthetic California", treatment_label="Proposal 99")

print(data)

print(dsc.original_data.weight_df)
print(dsc.original_data.comparison_df)
print(dsc.original_data.pen)

'''
#Plot validity tests
dsc.in_space_placebo()
#np.savetxt("pre_post_rmspe_ratio.csv", synth.pre_post_rmspe_ratio, delimiter=",")
dsc.plot(['rmspe ratio'])

dsc.in_time_placebo(5)
dsc.plot(['in-time placebo'])
'''
