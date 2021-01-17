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
import cvxpy as cvx
from scipy.optimize import minimize, differential_evolution

class Inferences(object):
    
    def optimize(self,
                treated_outcome, treated_covariates,
                control_outcome, control_covariates, 
                data,
                placebo,
                steps=8, verbose=False):
        '''
        Solves the nested optimization function of finding the optimal synthetic control

        placebo: bool
            indicates whether the optimization is ran for finding the real synthetic control
            or as part of a placebo-style validity test. If True, only placebo class attributes are affected.

        steps: int
            The number of different initializations of v_0 the gradient descent optimization is ran for
            Higher values mean longer running time but higher chances of finding a globally optimal solution

        verbose: bool, default=False
            If true, prints additional detail regarding the state of the optimization
        '''
        args = (treated_outcome, treated_covariates,
                control_outcome, control_covariates,
                placebo, data)
        
        for step in range(steps):

            #Dirichlet distribution returns a valid pmf over n_covariates states
            #v_0 = np.random.dirichlet(np.ones(data.n_covariates), size=1)
            v_0 = np.full(data.n_covariates, 1/data.n_covariates)

            #Required to have non negative values
            bnds = tuple((0,1) for _ in range(data.n_covariates))
            
            #Optimze
            res = minimize(self.total_loss, v_0,  args=(args),
                            method='L-BFGS-B', bounds=bnds, 
                            options={'gtol': 1e-6,'disp':3, 'iprint':3})
            
            if verbose:
                print("Successful:", res.success)
                print(res.message)
        
        #If sampler did not converge, try again up to times before admitting defeat
        try:
            res.x
        except:
            data.fail_count += 1
            if data.fail_count <= 1:
                data.optimize(*args)
            
        if self.method == "DSC":
            self._update_original_data(placebo)

        return

    def total_loss(self, v_0, 
                    treated_outcome, treated_covariates,
                    control_outcome, control_covariates, 
                    placebo, data):
        '''
        Solves for w*(v) that minimizes loss function 1 given v,
        Returns loss from loss function 2 with w=w*(v)

        placebo: bool
            indicates whether the optimization is ran for finding the real synthetic control
            or as part of a placebo-style validity test. If True, only placebo class attributes are affected.
        '''

        assert placebo in [False, "in-time", "in-space"], "TypeError: Placebo must False, 'in-time' or 'in-space'"

        n_controls = control_outcome.shape[1]
        
        V = np.zeros(shape=(data.n_covariates, data.n_covariates))
        np.fill_diagonal(V, v_0)
        
        # Construct the problem - constrain weights to be non-negative
        w = cvx.Variable((n_controls, 1), nonneg=True)
        
        #Define the objective
        objective = cvx.Minimize(cvx.sum(V @ cvx.square(treated_covariates - control_covariates @ w)))
        
        #Add constraint sum of weights must equal one
        constraints = [cvx.sum(w) == 1]
        
        #Solve problem
        problem = cvx.Problem(objective, constraints)
        
        try: #Try solving using current value of V, if it doesn't work return infinite loss
            result = problem.solve(verbose=False)
            loss = (treated_outcome - control_outcome @ w.value).T @ (treated_outcome - control_outcome @ w.value)
        except:
            return float(np.inf)
       
        #If loss is smaller than previous minimum, update loss, w and v
        if not placebo:
            if loss < data.min_loss:
                data.min_loss = loss
                data.w = w.value
                data.v = v_0
                data.synth_outcome = data.w.T @ data.control_outcome_all.T #Transpose to make it (n_periods x 1)
                data.synth_covariates = data.control_covariates @ data.w
        
        elif placebo == "in-space":
            data.in_space_placebo_w = w.value

        elif placebo == "in-time":
            data.in_time_placebo_w = w.value
            
        #Return loss
        return loss


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

            #Solve for best synthetic control weights
            self.optimize(treated_placebo_outcome[:data.periods_pre_treatment], 
                            treated_placebo_covariates,
                            control_placebo_outcome[:data.periods_pre_treatment], 
                            control_placebo_covariates,
                            data,
                            "in-space", n_optim)
            
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

        #Run find synthetic control from shortened pre-treatment period 
        self.optimize(in_time_placebo_treated_outcome, in_time_placebo_treated_covariates,
                        in_time_placebo_control_outcome, in_time_placebo_control_covariates,
                        data,
                        "in-time", n_optim)
        #Compute placebo outcomes using placebo_w vector from optimize
        placebo_outcome = data.in_time_placebo_w.T @ in_time_placebo_control_outcome_all.T

        #Store relevant results as class attributes, for plotting and retrieval
        data.placebo_treatment_period = placebo_treatment_period
        data.placebo_periods_pre_treatment = periods_pre_treatment
        data.in_time_placebo_outcome = placebo_outcome

        #Update original data if DSC
        if self.method == "DSC":
            self._update_original_data('in-time')

    
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

    def _pre_post_rmspe_ratios(self, placebo_outcomes, placebo=True):
        '''
        Computes the pre-post root mean square prediction error for all
        in-place placebos and the treated units
        '''
        data = self.original_data

        #Initialize ratio list with treated unit
        post_ratio_list, pre_ratio_list = [], []

        #Add treated unit
        if not placebo:
            #Compute rmspe
            treated_post, treated_pre = self._pre_post_rmspe(data.synth_outcome.T, data.treated_outcome_all)
            post_ratio_list.append(treated_post)
            pre_ratio_list.append(treated_pre)

            #Store in dataframe
            rmspe_df = pd.DataFrame({"unit": data.treated_unit,
                                    "pre_rmspe": pre_ratio_list,
                                    "post_rmspe": post_ratio_list},
                                    columns=["unit", "pre_rmspe", "post_rmspe"])
            #Compute post/pre rmspe ratio
            rmspe_df["post/pre"] = rmspe_df["post_rmspe"] / rmspe_df["pre_rmspe"]

            data.rmspe_df = rmspe_df
            return
            

        else: #if placebo
            #Add each control unit and respective synthetic control
            for i in range(data.n_controls):
                post_ratio, pre_ratio = self._pre_post_rmspe(placebo_outcomes[i], data.control_outcome_all[:, i].T, placebo=True)
                post_ratio_list.append(post_ratio)
                pre_ratio_list.append(pre_ratio)

            #Store in dataframe
            rmspe_df = pd.DataFrame({"unit":  data.control_units,
                                    "pre_rmspe": pre_ratio_list,
                                    "post_rmspe": post_ratio_list},
                                    columns=["unit", "pre_rmspe", "post_rmspe"])
            
            #Compute post/pre rmspe ratio
            rmspe_df["post/pre"] = rmspe_df["post_rmspe"] / rmspe_df["pre_rmspe"]
            
            #Extend self.original_data.rmspe_df and return
            rmspe_df = pd.concat([data.rmspe_df, rmspe_df], axis=0)
            data.rmspe_df = rmspe_df.reset_index(drop=True)
            return
             

    def _pre_post_rmspe(self, synth_outcome, treated_outcome, placebo=False):
        '''
        Input: Takes two outcome time series of the same dimensions

        Returns:
        post-treatment root mean square prediction error
        and
        pre-treatment root mean square prediction error
        '''

        t = self.original_data.periods_pre_treatment

        if not placebo:
            pre_treatment = np.sqrt(((treated_outcome[:t] - synth_outcome[:t]) ** 2).mean())
    
            post_treatment = np.sqrt(((treated_outcome[t:] - synth_outcome[t:]) ** 2).mean())

        else:
            pre_treatment = np.sqrt(((treated_outcome[:t] - synth_outcome[0][:t]) ** 2).mean())
    
            post_treatment = np.sqrt(((treated_outcome[t:] - synth_outcome[0][t:]) ** 2).mean())


        return post_treatment, pre_treatment

    def _update_original_data(self, placebo):
        '''
        Used only in DiffSynth / DSC:

        Called at the end of optimization procedure:

        Transcribes relevant results from ModifiedData to OriginalData
        '''
        if not placebo:
            self.original_data.w = self.modified_data.w
            self.original_data.v = self.modified_data.v
            self.original_data.synth_constant, self.original_data.synth_outcome = self._get_dsc_outcome(self.original_data.w,
                                                                                                        self.original_data.control_outcome_all,
                                                                                                        self.original_data.periods_pre_treatment,
                                                                                                        self.original_data.treated_outcome)
        elif placebo == 'in-space':
            self.original_data.in_space_placebo_w = self.modified_data.in_space_placebo_w
            self.original_data.pre_post_rmspe_ratio = self.modified_data.pre_post_rmspe_ratio 
            self.original_data.in_space_placebos = self.modified_data.in_space_placebos

        else: #Update in-time placebo
            self.original_data.placebo_treatment_period = self.modified_data.placebo_treatment_period
            self.original_data.placebo_periods_pre_treatment = self.modified_data.placebo_periods_pre_treatment
            self.original_data.in_time_placebo_w = self.modified_data.in_time_placebo_w
            _, self.original_data.in_time_placebo_outcome = self._get_dsc_outcome(self.original_data.in_time_placebo_w,
                                                                                self.original_data.control_outcome_all,
                                                                                self.original_data.placebo_periods_pre_treatment,
                                                                                self.original_data.treated_outcome[:self.original_data.placebo_periods_pre_treatment])
        return 

    def _get_dsc_outcome(self, w, control_outcome, periods_pre_treatment, treated_pretreatment_outcome):
        '''Method used only by DiffSynth (DSC)
        
        Arguments:

          w: np.array
            Weight matrix (n_controls x 1)

          control_outcome: np.array
            Outcome matrix for all control units for all time periods (n_controls x n_periods_all)
          
          periods_pre_treatment: int
            Integer representing the number of periods before treatment

          treated_pretreatment_outcome: np.array
            Outcome matrix for treated unit (1 x n_periods_pre_treatment)

        Approach:
        1. Solve for the differenced synthetic control, less the constant
        2. Solve for the constant by computing the average difference, in the pre-treatment period, 
           between the treated unit and (1.)
        3. Add the constant to all time periods in (1). This is the outcome of the differenced synthtic control.
        '''
        #1. Compute synthetic control outcome, less constant
        synth_outcome =  w.T @ control_outcome.T
        synth_outcome_pre_treatment = w.T @ control_outcome[:periods_pre_treatment].T

        #2. Constant defined to be average difference between synth and treated unit in the pre-treatment period
        constant = np.mean(treated_pretreatment_outcome - synth_outcome_pre_treatment)

        #3. Add constant to synthetic control outcome
        synth_outcome += constant

        return constant, synth_outcome
    
    ##########################################################
    ## ALTERNATE OPTIMIZATION METHODS -- NOT CURRENTLY USED ##
    ##########################################################

    def diffevo_optimize(self):
        '''Uses the differential evolution optimizer from scipy to solve for synthetic control'''
        bounds = [(0,1) for _ in range(self.n_covariates)]

        result = differential_evolution(self.total_loss, bounds)
        
        self.v = result.x
        
        self.w, loss = self.total_loss(self.v, False)
        
        return self.w, loss


    def random_optimize(self, steps=10**4):
        '''
        "When intelligent approaches fail, throw spaghetti at the wall and see what sticks" - Benito Mussolini
        
        The below random samples valid v matrices from a dirichlet distribution,
        then computes the resulting w*(v) and the total loss associated with it
        
        Returns the w*(v) that minimizes total loss, and the total loss
        '''
        #Initalize variable to track best w*(v)
        best_w, min_loss = None, float(np.inf)
        for i in range(steps):
            
            #Generate sample v
            #Dirichlet distribution returns a valid pmf over n_covariates states
            v = np.random.dirichlet(np.ones(self.n_covariates), size=1)
            
            #Print progress
            if (i+1)%steps/10 == 0:
                print('{}%'.format((i+1)%steps/10))
            
            #Compute w*(v) and loss for v
            w, loss = self.total_loss(v, False)
            
            #See if w*(v) results in lower loss, if so update best
            if loss < min_loss:
                best_w, min_loss = w, loss
        
        
        #Store, print, return best solutions
        self.w = best_w
        return best_w, min_loss