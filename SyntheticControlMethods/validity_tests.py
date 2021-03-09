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

import numpy as np
import pandas as pd

class ValidityTests(object):
    '''
    This class contains all validity tests for evaluating synthetic controls.
    This includes:

    in_space_placebos

    in_time_placebos

    rmspe_df

    weight_df

    comparison_df
    '''

    def in_space_placebo(self, n_optim=3):
        '''
        Fits a synthetic control to each of the control units, 
        using the remaining control units as control group

        Returns:
            matrix (n_controls x n_periods) with the outcome for each synthetic control

        procedure:
        1. find way to remove one control unit from control_covariate and control_outcome
        2. For loop over all control units, fitting a control to each
        3. Store the outcome values for each synthetic control
        '''
        #See which instance of SynthData to use depending on if we are using DiffSynth or Synth
        data = self.original_data if self.method=='SC' else self.modified_data

        placebo_outcomes = []
        for i in range(data.n_controls):
            #Format placebo and control data
            treated_placebo_outcome = data.control_outcome_all[:,i].reshape(data.periods_all, 1)

            treated_placebo_covariates = data.control_covariates[:,i].reshape(data.n_covariates, 1)

            control_placebo_outcome = np.array([data.control_outcome_all[:,j] for j in range(data.n_controls) if j != i]).T
            control_placebo_covariates = np.array([[data.control_covariates[x,j] for j in range(data.n_controls) if j != i] for x in range(data.n_covariates)])

            #Rescale covariates to be unit variance (helps with optimization)
            treated_placebo_covariates, control_placebo_covariates = self._rescale_covariate_variance(treated_placebo_covariates,
                                                                            control_placebo_covariates,
                                                                            data.n_covariates)

            pairwise_difference = treated_placebo_covariates - control_placebo_covariates

            #Solve for best synthetic control weights
            self.optimize(treated_placebo_outcome[:data.periods_pre_treatment], 
                            treated_placebo_covariates,
                            control_placebo_outcome[:data.periods_pre_treatment], 
                            control_placebo_covariates,
                            pairwise_difference,
                            data,
                            "in-space", "auto", n_optim)
            
            #Compute outcome of best synthetic control
            if self.method == "SC":
                synthetic_placebo_outcome = data.in_space_placebo_w.T @ control_placebo_outcome.T

            else: #If method == 'DSC'
                _, synthetic_placebo_outcome = self._get_dsc_outcome(data.in_space_placebo_w,
                                                                np.array([self.original_data.control_outcome_all[:,j] for j in range(data.n_controls) if j != i]).T,
                                                                data.periods_pre_treatment,
                                                                self.original_data.control_outcome[:,i].reshape(data.periods_pre_treatment+1, 1))

            #Store it
            placebo_outcomes.append(synthetic_placebo_outcome)
        
        #Compute pre-post RMSPE Ratio
        data.pre_post_rmspe_ratio = self._pre_post_rmspe_ratios(placebo_outcomes)
        data.in_space_placebos = self._normalize_placebos(placebo_outcomes)
        
        if self.method == "DSC":
            self._update_original_data('in-space')

        return

    def in_time_placebo(self, placebo_treatment_period, n_optim=5):
        '''
        Fits a synthetic control to the treated unit,
        with a pre-treatment period shorter than the true pre-treatment period,
        i.e. telling the algorithm the treatment took place before the true treatment period

        Interpretation: we expect the treatment effect to be small in the "post-treatment periods" that pre-date the true treatment

        Returns:
            (1 x n_periods) matrix with the outcome for in-time placebo
        '''
        data = self.original_data if self.method=='SC' else self.modified_data

        periods_pre_treatment = data.dataset.loc[data.dataset[data.time]<placebo_treatment_period][data.time].nunique()

        #Format necessary matrices, but do so with the new, earlier treatment period
        ###Get treated unit matrices first###
        in_time_placebo_treated_outcome_all, in_time_placebo_treated_outcome, in_time_placebo_treated_covariates = self._process_treated_data(
            data.dataset, data.outcome_var, data.id, data.time, 
            placebo_treatment_period, data.treated_unit, data.periods_all, 
            periods_pre_treatment, data.covariates, data.n_covariates
        )

        ### Now for control unit matrices ###
        in_time_placebo_control_outcome_all, in_time_placebo_control_outcome, in_time_placebo_control_covariates = self._process_control_data(
            data.dataset, data.outcome_var, data.id, data.time, 
            placebo_treatment_period, data.treated_unit, data.n_controls, 
            data.periods_all, periods_pre_treatment, data.covariates
        )

        pairwise_difference = in_time_placebo_treated_covariates - in_time_placebo_control_covariates

        #in_time_placebo_treated_covariates = in_time_placebo_treated_covariates.reshape(1, data.n_covariates)
        #Run find synthetic control from shortened pre-treatment period 
        self.optimize(in_time_placebo_treated_outcome, in_time_placebo_treated_covariates,
                        in_time_placebo_control_outcome, in_time_placebo_control_covariates,
                        pairwise_difference,
                        data,
                        "in-time", 0, n_optim)

        #Compute placebo outcomes using placebo_w vector from optimize
        placebo_outcome = data.in_time_placebo_w.T @ in_time_placebo_control_outcome_all.T

        #Store relevant results as class attributes, for plotting and retrieval
        data.placebo_treatment_period = placebo_treatment_period
        data.placebo_periods_pre_treatment = periods_pre_treatment
        data.in_time_placebo_outcome = placebo_outcome

        #Update original data if DSC
        if self.method == "DSC":
            self._update_original_data('in-time')

    def _normalize_placebos(self, placebo_outcomes):
        '''
        Takes array of synthetic placebo outcomes

        returns array of same dimension where the control unit outcome has been subtracted from
        the synthetic placebo
        '''
        data = self.original_data

        #Initialize ratio list with treated unit
        normalized_placebo_outcomes = []

        #Add each control unit and respective synthetic control
        for i in range(data.n_controls):
            normalized_outcome = (placebo_outcomes[i] - data.control_outcome_all[:, i].T).T
            normalized_placebo_outcomes.append(normalized_outcome)

        return normalized_placebo_outcomes

    def predictor_table(self, 
                        treated_label="Treated Unit", 
                        synth_label="Synthetic Control",
                        include_donor_pool_average=False,
                        donor_pool_label="Donor pool"):
        '''
        Returns a Dataframe with the mean of each predictor for the treated vs control
        Rows: predictors
        Columns: Treated unit, Synthetic Control, Donor pool average (optional)
        
        include_donor_pool_average: bool. Default=False.
        Whether or not include a third column with the simple average for the entire donor pool.
        '''

        data = self.original_data

        if not include_donor_pool_average:
            
            table = pd.DataFrame({treated_label: data.treated_covariates.ravel(), 
                                 synth_label: (data.synth_covariates).ravel()},
                                index=data.covariates)
        
        else:
            
            table = pd.DataFrame({treated_label: data.treated_covariates.ravel(), 
                                 synth_label: (data.synth_covariates).ravel(),
                                 donor_pool_label: data.control_covariates.mean(axis=1)},
                                index=data.covariates)
        return table