

#Import packages
import pandas as pd
import numpy as np
from SyntheticControlMethods import Synth, DiffSynth

#Import data
data = pd.read_csv("/Users/oscarengelbrektson/Documents/test_dataset.csv")

#Fit Synthetic Control
sc = DiffSynth(data, "y", "ID", "Time", 10, "A", n_optim=30)


sc.plot(["original", "pointwise", "cumulative"], treated_label="California", 
            synth_label="Synthetic California", treatment_label="Proposal 99")

#Plot validity tests
sc.in_space_placebo()
#np.savetxt("pre_post_rmspe_ratio.csv", synth.pre_post_rmspe_ratio, delimiter=",")
sc.plot(['rmspe ratio'], in_space_exclusion_multiple=None)