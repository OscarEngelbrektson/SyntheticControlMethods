# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import copy

from SyntheticControlMethods.plot import Plot
from SyntheticControlMethods.inferences import Inferences

class SynthBase(object):
    
    def __init__(self, dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, control_units,
                covariates, periods_all, periods_pre_treatment, n_controls, n_covariates,
                treated_outcome, control_outcome, treated_covariates, control_covariates, 
                unscaled_treated_covariates, unscaled_control_covariates,
                treated_outcome_all, control_outcome_all, pairwise_difference, pen,
                w=None, v=None, **kwargs):

        '''
        INPUT VARIABLES:
        
        dataset: the dataset for the synthetic control procedure.
        Should have the the following column structure:
        ID, Time, outcome_var, x0, x1,..., xn
        Each row in dataset represents one observation.
        The dataset should be sorted on ID then Time. 
        That is, all observations for one unit in order of Time, 
        followed by all observations by the next unit also sorted on time
        
        ID: a string containing a unique identifier for the unit associated with the observation.
        E.g. in the simulated datasets provided, the ID of the treated unit is "A".
        
        Time: an integer indicating the time period to which the observation corresponds.
        
        treated_unit: ID of the treated unit
        '''

        self.dataset = dataset
        self.outcome_var = outcome_var
        self.id = id_var
        self.time = time_var
        self.treatment_period = treatment_period
        self.treated_unit = treated_unit
        self.control_units = control_units
        self.covariates = covariates
        self.periods_all = periods_all
        self.periods_pre_treatment = periods_pre_treatment
        self.n_controls = n_controls
        self.n_covariates = n_covariates
        self.pen = pen

        
        '''
        PROCESSED VARIABLES:
        
        treated_outcome: a (1 x treatment_period) matrix containing the
        outcome of the treated unit for each observation in the pre-treatment period.
        Referred to as Z1 in Abadie, Diamond, Hainmueller.
        
        control_outcome: a ((len(unit_list)-1) x treatment_period) matrix containing the
        outcome of every control unit for each observation in the pre-treatment period
        Referred to as Z0 in Abadie, Diamond, Hainmueller.
        
        treated_outcome_all: a (1 x len(time)) matrix
        same as treated_outcome but includes all observations, including post-treatment
        
        control_outcome_all: a (n_controls x len(time)) matrix
        same as control_outcome but includes all observations, including post-treatment
        
        treated_covariates: a (1 x len(covariates)) matrix containing the
        average value for each predictor of the treated unit in the pre-treatment period
        Referred to as X1 in Abadie, Diamond, Hainmueller.
        
        control_covariates: a (n_controls x len(covariates)) matrix containing the
        average value for each predictor of every control unit in the pre-treatment period
        Referred to as X0 in Abadie, Diamond, Hainmueller.
        
        W: a (1 x n_controls) matrix containing the weights assigned to each
        control unit in the synthetic control. W is contstrained to be convex,
        that is sum(W)==1 and ∀w∈W, w≥0, each weight is non-negative and all weights sum to one.
        Referred to as W in Abadie, Diamond, Hainmueller.
        
        V: a (len(covariates) x len(covariates)) matrix representing the relative importance
        of each covariate. V is contrained to be diagonal, positive semi-definite. 
        Pracitcally, this means that the product V.control_covariates and V.treated_covariates
        will always be non-negative. Further, we constrain sum(V)==1, otherwise there will an infinite
        number of solutions V*c, where c is a scalar, that assign equal relative importance to each covariate
        Referred to as V in Abadie, Diamond, Hainmueller.
        '''
        ###Post processing quantities
        self.treated_outcome = treated_outcome
        self.control_outcome = control_outcome
        self.treated_covariates = treated_covariates
        self.control_covariates = control_covariates
        self.unscaled_treated_covariates = unscaled_treated_covariates
        self.unscaled_control_covariates = unscaled_control_covariates
        self.treated_outcome_all = treated_outcome_all
        self.control_outcome_all = control_outcome_all
        self.pairwise_difference = pairwise_difference

        ###Post inference quantities
        self.w = w #Can be provided
        self.v = v #Can be provided
        self.weight_df = None
        self.comparison_df = None
        self.synth_outcome = None
        self.synth_constant = None
        self.synth_covariates = None
        self.rmspe_df = None
        #used in optimization
        self.min_loss = float("inf")
        self.fail_count = 0 #Used to limit number of optimization attempts

        ###Validity tests
        self.in_space_placebos = None
        self.in_space_placebo_w = None
        self.pre_post_rmspe_ratio = None
        self.in_time_placebo_outcome = None
        self.in_time_placebo_treated_outcome = None
        self.in_time_placebo_w = None
        self.placebo_treatment_period = None
        self.placebo_periods_pre_treatment = None

class DataProcessor(object):
    
    def _process_input_data(self, dataset, 
                            outcome_var, id_var, time_var, 
                            treatment_period, treated_unit, 
                            pen, exclude_columns, 
                            **kwargs):
        '''
        Extracts processed variables, excluding v and w, from input variables.
        These are all the data matrices.
        '''
        #All columns not y, id or time must be predictors
        covariates = [col for col in dataset.columns if col not in [id_var, time_var] and col not in exclude_columns]

        #Extract quantities needed for pre-processing matrices
        #Get number of periods in pre-treatment and total
        periods_all = dataset[time_var].nunique()
        periods_pre_treatment = dataset.loc[dataset[time_var]<treatment_period][time_var].nunique()
        #Number of control units, -1 to remove treated unit
        n_controls = dataset[id_var].nunique() - 1
        n_covariates = len(covariates)

        #All units that are not the treated unit are controls
        control_units = dataset.loc[dataset[id_var] != treated_unit][id_var].unique()

        ###Get treated unit matrices first###
        treated_outcome_all, treated_outcome, unscaled_treated_covariates = self._process_treated_data(
            dataset, outcome_var, id_var, time_var, 
            treatment_period, treated_unit, periods_all, 
            periods_pre_treatment, covariates, n_covariates
        )

        ### Now for control unit matrices ###
        control_outcome_all, control_outcome, unscaled_control_covariates = self._process_control_data(
            dataset, outcome_var, id_var, time_var, 
            treatment_period, treated_unit, n_controls, 
            periods_all, periods_pre_treatment, covariates
        )
        
        #Rescale covariates to be unit variance (helps with optimization)
        treated_covariates, control_covariates = self._rescale_covariate_variance(unscaled_treated_covariates,
                                                                            unscaled_control_covariates,
                                                                            n_covariates)
        
        #Get matrix of unitwise differences between control units to treated unit
        pairwise_difference = self._get_pairwise_difference_matrix(treated_covariates, 
                                                                   control_covariates)

        return {
            'dataset': dataset,
            'outcome_var':outcome_var,
            'id_var':id_var,
            'time_var':time_var,
            'treatment_period':treatment_period,
            'treated_unit':treated_unit,
            'control_units':control_units,
            'covariates':covariates,
            'periods_all':periods_all,
            'periods_pre_treatment':periods_pre_treatment,
            'n_controls': n_controls,
            'n_covariates':n_covariates,
            'treated_outcome_all': treated_outcome_all,
            'treated_outcome': treated_outcome,
            'treated_covariates': treated_covariates,
            'unscaled_treated_covariates':unscaled_treated_covariates,
            'control_outcome_all': control_outcome_all,
            'control_outcome': control_outcome,
            'control_covariates': control_covariates,
            'unscaled_control_covariates': unscaled_control_covariates,
            'pairwise_difference':pairwise_difference,
            'pen':pen
        }
    
    def _process_treated_data(self, dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, 
                            periods_all, periods_pre_treatment, covariates, n_covariates):
        '''
        Extracts and formats outcome and covariate matrices for the treated unit
        '''

        treated_data_all = dataset.loc[dataset[id_var] == treated_unit]
        treated_outcome_all = np.array(treated_data_all[outcome_var]).reshape(periods_all,1) #All outcomes
        
        #Only pre-treatment
        treated_data = treated_data_all.loc[dataset[time_var] < treatment_period]
        #Extract outcome and shape as matrix
        treated_outcome = np.array(treated_data[outcome_var]).reshape(periods_pre_treatment, 1)
        #Columnwise mean of each covariate in pre-treatment period for treated unit, shape as matrix
        treated_covariates = np.array(treated_data[covariates].mean(axis=0)).reshape(n_covariates, 1)

        return treated_outcome_all, treated_outcome, treated_covariates
    

    def _process_control_data(self, dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, n_controls, 
                            periods_all, periods_pre_treatment, covariates):
        '''
        Extracts and formats outcome and covariate matrices for the control group
        '''

        #Every unit that is not the treated unit is control
        control_data_all = dataset.loc[dataset[id_var] != treated_unit]
        control_outcome_all = np.array(control_data_all[outcome_var]).reshape(n_controls, periods_all).T #All outcomes
        
        #Only pre-treatment
        control_data = control_data_all.loc[dataset[time_var] < treatment_period]
        #Extract outcome, then shape as matrix
        control_outcome = np.array(control_data[outcome_var]).reshape(n_controls, periods_pre_treatment).T
        
        #Extract the covariates for all the control units
        #Identify which rows correspond to which control unit by setting index,
        #then take the unitwise mean of each covariate
        #This results in the desired (n_control x n_covariates) matrix
        control_covariates = np.array(control_data[covariates].\
                set_index(np.arange(len(control_data[covariates])) // periods_pre_treatment).mean(level=0)).T

        return control_outcome_all, control_outcome, control_covariates

    def _rescale_covariate_variance(self, treated_covariates, control_covariates, n_covariates):
        '''Rescale covariates to be unit variance'''

        #Combine control and treated into one big dataframe, over which we will compute variance for each covariate
        big_dataframe = np.concatenate((treated_covariates, control_covariates), axis=1)

        #Rescale each covariate to have unit variance
        big_dataframe /= np.apply_along_axis(np.std, 0, big_dataframe)

        #Re-seperate treated and control from big dataframe
        treated_covariates = big_dataframe[:,0].reshape(n_covariates, 1) #First column is treated unit
        control_covariates = big_dataframe[:,1:] #All other columns are control units

        #Return covariate matices with unit variance
        return treated_covariates, control_covariates
    
    def _get_pairwise_difference_matrix(self, treated_covariates, control_covariates):
        '''
        Computes matrix of same shape as control_covariates, but with unit-wise difference from treated unit
        Used in optimization objective for both SC and DSC
        '''
        return treated_covariates - control_covariates
    
    def _get_weight_df(self, data):
      '''Prepares dataframe with weight assigned to each unit in synthetic control'''
      
      weight_df = pd.DataFrame({"Unit":data.control_units,
                                        "Weight":data.w.T[0]})
      #Show only units with non-zero weights
      return weight_df.loc[weight_df["Weight"] > 0.00001] 
    
    def _get_comparison_df(self, data):
      '''
      Returns dataframe with shape (n_covariates, 4).
      The four columns are:

        self.original_data.treated_unit: 
          Unscaled, average covariate values of the treated unit
          If method == DSC, then the differenced data is displayed instead
        
        Synthetic Control: 
          Unscaled, covariate values of the synthetic control unit
          If method == DSC, then the differenced data is displayed instead
      
        WMAUE:
          Weighted Mean Absolute Unitwise Error. For each covariate, how different is each control 
          unit inside the synthetic control from the treated unit, weighted by the weight assigned to each unit.
          This does not change even if method == DSC, as bias scales with value of difference and not change
        
        
        Importance:
          Leading diagonal of V matrix. How important, relative to other covariates,
          is matching on each covariate in the optimization process?
          Note that this is computed after rescaling each covariate to be unit variance, 
          whereas the other columns show the unscaled covariate values.
      
      Interpretation:
      If the synthetic control has good fit, the following things should be true:

        1. Each row of the first two columns should be approximately equal. 
           This means the synthetic control has reconstructed the treated unit values.

        2. The third column should be small, relative to the values in columns 1 and 2. 
           The closer to zero, the more similar the individual control units inside the syntetic control are to the treated unit.
           The smaller the MWAUE, the lower the potential bias, all else equal.
        
        3. There is no fixed way to interpret the importance column. Instead, it should be evaluated using domain knowledge.
           Is the relative importance assigned to each covariate reasonable given the context?
      '''
      data = self.original_data if self.method=='SC' else self.modified_data

      #WMAPE
      unscaled_treated_covariates = self.original_data.unscaled_treated_covariates 
      unscaled_control_covariates = self.original_data.unscaled_control_covariates
      wmape = (np.abs((unscaled_treated_covariates - unscaled_control_covariates)) @ data.w).reshape(data.n_covariates,)
      
      comparison_df = pd.DataFrame({data.treated_unit: data.unscaled_treated_covariates.ravel(),
                                    "Synthetic " + data.treated_unit: (data.unscaled_control_covariates @ data.w).ravel(),
                                    "WMAPE": wmape,
                                    "Importance":data.v},
                                    index=data.covariates)
      return comparison_df
      

class Synth(Inferences, Plot, DataProcessor):

    def __init__(self, dataset, 
                outcome_var, id_var, time_var, 
                treatment_period, treated_unit, 
                n_optim=10, pen=0, exclude_columns=[],
                **kwargs):
        '''
        data: 
          Type: Pandas dataframe. 
          A pandas dataframe containing the dataset. Each row should contain one observation for a unit at a time, 
          including the outcome and covariates. Dataset should be ordered by unit then time.

        outcome_var: 
          Type: str. 
          Name of outcome column in data, e.g. "gdp"

        id_var: 
          Type: str. 
          Name of unit indicator column in data, e.g. "country"

        time_var: 
          Type: str. 
          Name of time column in data, e.g. "year"

        treatment_period: 
          Type: int. 
          Time of first observation after the treatment took place, i.e. first observation affected by the treatment effect.
          E.g. 1990 for german reunification.

        treated_unit: 
          Type: str. 
          Name of the unit that recieved treatment, e.g. "West Germany"
          data["id_var"] == treated_unit

        n_optim: 
          Type: int. Default: 10. 
          Number of different initialization values for which the optimization is run. 
          Higher number means longer runtime, but a higher chance of a globally optimal solution.

        pen:
          Type: float. Default: 0.
          Penalization coefficient which determines the relative importance of minimizing the sum of the pairwise difference of each individual
          control unit in the synthetic control and the treated unit, vis-a-vis the difference between the synthetic control and the treated unit.
          Higher number means pairwise difference matters more.
        '''
        self.method = "SC"

        original_checked_input = self._process_input_data(
            dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, pen, exclude_columns, **kwargs
        )
        self.original_data = SynthBase(**original_checked_input)

        #Get synthetic Control
        self.optimize(self.original_data.treated_outcome, self.original_data.treated_covariates,
                    self.original_data.control_outcome, self.original_data.control_covariates,
                    self.original_data.pairwise_difference,
                    self.original_data, False, pen, n_optim)
        
        #Compute rmspe_df
        self._pre_post_rmspe_ratios(None, False)

        #Prepare weight_df with unit weights
        self.original_data.weight_df = self._get_weight_df(self.original_data)
        self.original_data.comparison_df = self._get_comparison_df(self.original_data)


class DiffSynth(Inferences, Plot, DataProcessor):

    def __init__(self, dataset, 
                outcome_var, id_var, time_var, 
                treatment_period, treated_unit, 
                n_optim=10, pen=0, 
                exclude_columns=[], not_diff_cols=None,
                **kwargs):
        '''
        data: 
          Type: Pandas dataframe. 
          A pandas dataframe containing the dataset. Each row should contain one observation for a unit at a time, 
          including the outcome and covariates. Dataset should be ordered by unit then time.

        outcome_var: 
          Type: str. 
          Name of outcome column in data, e.g. "gdp"

        id_var: 
          Type: str. 
          Name of unit indicator column in data, e.g. "country"

        time_var: 
          Type: str. 
          Name of time column in data, e.g. "year"

        treatment_period: 
          Type: int. 
          Time of first observation after the treatment took place, i.e. first observation affected by the treatment effect.
          E.g. 1990 for german reunification.

        treated_unit: 
          Type: str. 
          Name of the unit that recieved treatment, e.g. "West Germany"
          data["id_var"] == treated_unit

        n_optim: 
          Type: int. Default: 10. 
          Number of different initialization values for which the optimization is run. 
          Higher number means longer runtime, but a higher chance of a globally optimal solution.

        not_diff_cols:
            Type: list. Default: [].
            List of column names to omit from pre-processing, e.g. compute the first difference for. 
            Typically, columns should be included if the proportion of missing values is high.
            This is because the first difference is only defined for two consecutive values.
         
        pen:
         Type: float. Default: 0.
         Penalization coefficient which determines the relative importance of minimizing the sum of the pairwise difference of each individual
         control unit in the synthetic control and the treated unit, vis-a-vis the difference between the synthetic control and the treated unit.
         Higher number means pairwise difference matters more.
        '''
        self.method = "DSC"

        #Process original data - will be used in plotting and summary
        original_checked_input = self._process_input_data(
            dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, pen, **kwargs
        )
        self.original_data = SynthBase(**original_checked_input)

        #Process differenced data - will be used in inference and optimization
        modified_dataset = self.difference_data(dataset, not_diff_cols)
        modified_checked_input = self._process_input_data(
            modified_dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, pen, **kwargs
        )
        self.modified_data = SynthBase(**modified_checked_input)
        self.modified_data.pairwise_difference = self.original_data.pairwise_difference 

        #Get synthetic Control
        self.optimize(self.modified_data.treated_outcome, self.modified_data.treated_covariates,
                    self.modified_data.control_outcome, self.modified_data.control_covariates,
                    self.modified_data.pairwise_difference,
                    self.modified_data, False, pen, n_optim)
        
        #Compute rmspe_df for treated unit Synthetic Control
        self._pre_post_rmspe_ratios(None, False)

        #Prepare summary tables
        self.original_data.weight_df = self._get_weight_df(self.original_data)
        self.original_data.comparison_df = self._get_comparison_df(self.original_data)

    def difference_data(self, dataset, not_diff_cols):
        '''
        Takes an appropriately formatted, unprocessed dataset
        returns dataset with first-difference values (change from previous time period) 
        computed unitwise for the outcome and all covariates
        Ready to fit a Differenced Synthetic Control

        Transformation method - First Differencing: 
        
        Additional processing:

        1. Imputes missing values using linear interpolation. 
           An important step because the first difference is undefined if two consecutive periods are not present.
        '''
        #Make deepcopy of original data as base
        modified_dataset = copy.deepcopy(dataset)
        data = self.original_data

        #Binary flag for whether there are columns to ignore
        ignore_all_cols = not_diff_cols == None

        #Compute difference of outcome variable
        modified_dataset[data.outcome_var] = modified_dataset.groupby(data.id)[data.outcome_var].apply(lambda unit: unit.interpolate(method='linear', limit_direction="both")).diff()
        
        #For covariates
        for col in data.covariates:
            #Fill in missing values using unitwise linear interpolation
            modified_dataset[col] = modified_dataset.groupby(data.id)[col].apply(lambda unit: unit.interpolate(method='linear', limit_direction="both"))
            
            #Compute change from previous period
            if not ignore_all_cols:
                if col not in not_diff_cols:
                    modified_dataset[col].diff()

        #Drop first time period for every unit as the change from the previous period is undefined
        modified_dataset.drop(modified_dataset.loc[modified_dataset[data.time]==modified_dataset[data.time].min()].index, inplace=True)
        #Return resulting dataframe
        return modified_dataset
    
    def demean_data(self):
        '''
        Takes an appropriately formatted, unprocessed dataset
        returns dataset with demeaned values computed unitwise for the outcome and all covariates
        Ready to fit a Differenced Synthetic Control

        Transformation method - MeanSubtraction: 
        Subtracting the mean of the corresponding variable and unit from every observation
        '''
        raise NotImplementedError
        
        mean_subtract_cols = self.dataset.groupby(self.id).apply(lambda x: x - np.mean(x)).drop(columns=[self.time], axis=1)
        return pd.concat([data[["ID", "Time"]], mean_subtract_cols], axis=1)
