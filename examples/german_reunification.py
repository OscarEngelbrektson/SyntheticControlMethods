#Import packages
import pandas as pd
from synth import Synth

#Import data
data = pd.read_csv("examples/german_reunification.csv")
data = data.drop(columns="code", axis=1)

#Fit Synthetic Control
synth = Synth(data, "gdp", "country", "year", 1990, "West Germany")
print(synth.w, sum(synth.w))
synth.plot(["original", "pointwise", "cumulative"])