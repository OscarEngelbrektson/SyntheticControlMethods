#Import packages
import pandas as pd
import numpy as np
from SyntheticControlMethods import Synth, DiffSynth

#Import data
data_dir = "https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/master/examples/datasets/"
data = pd.read_csv(data_dir + "basque_data" + ".csv")
data = data.drop(columns=["regionno", "Unnamed: 0"], axis=1) #Drop superfluous columns
data = data.loc[data["regionname"] != "Spain (Espana)"] #Exclude spain as a valid control unit, as it includes basque, the treated unit.

#Fit Synthetic Control
sc = Synth(data, "gdpcap", "regionname", "year", 1970, "Basque Country (Pais Vasco)", n_optim=5, pen=0)

#Visualize
sc.plot(["original", "pointwise", "cumulative"], treated_label="Basque Country", 
            synth_label="Synthetic Basque Country", treatment_label="Terrorism")


print(sc.original_data.weight_df)
print(sc.original_data.comparison_df)
print(sc.original_data.pen)

'''
Placeholder: Insert validity tests
'''

### Repeat but with DSC
dsc = DiffSynth(data, "gdp", "country", "year", 1990, "West Germany", not_diff_cols=["schooling", "invest60", "invest70", "invest80"], n_optim=10, pen="auto")

#sc = DiffSynth(data, "gdp", "country", "year", 1990, "West Germany", not_diff_cols=["schooling", "invest60", "invest70", "invest80"], n_optim=1)
print(dsc.original_data.weight_df)
print(dsc.original_data.comparison_df)
print(dsc.original_data.pen)

#Fit
dsc.plot(["original", "pointwise", "cumulative"], treated_label="West Germany", 
            synth_label="Synthetic West Germany", treatment_label="German Reunification")


'''
Placeholder: Insert validity tests
'''