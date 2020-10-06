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

    def total_loss(self, v_0, 
                    treated_outcome, treated_covariates,
                    control_outcome, control_covariates, 
                    placebo):
        '''
        Solves for w*(v) that minimizes loss function 1 given v,
        Returns loss from loss function 2 with w=w*(v)
        '''
        assert type(placebo)==bool, "TypeError: Placebo must be True or False"

        n_controls = control_outcome.shape[1]
        
        V = np.zeros(shape=(self.n_covariates, self.n_covariates))
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
            if loss < self.min_loss:
                self.min_loss = loss
                self.w = w.value
                self.v = v_0
                self.synth_outcome = self.w.T @ self.control_outcome_all.T #Transpose to make it (n_periods x 1)
                self.synth_covariates = self.control_covariates @ self.w
        
        else:
            self.placebo_w = w.value
            
        #Return loss
        return loss

    
    def optimize(self,
                treated_outcome, treated_covariates,
                control_outcome, control_covariates,
                placebo,
                steps=8, verbose=False):
        '''
        
        '''
        args = (treated_outcome, treated_covariates,
                control_outcome, control_covariates,
                placebo)
        
        for step in range(steps):

            #Dirichlet distribution returns a valid pmf over n_covariates states
            v_0 = np.random.dirichlet(np.ones(self.n_covariates), size=1)

            #Required to have non negative values
            bnds = tuple((0,1) for _ in range(self.n_covariates))
            
            #Optimze
            res = minimize(self.total_loss, v_0,  args=(args),
                            method='L-BFGS-B', bounds=bnds, 
                            options={'gtol': 1e-6,'disp':3, 'iprint':3})
            
            if verbose:
                print("Successful:", res.success)
                print(res.message)
        
        #If sampler did not converge, try again up to 3 times before admitting defeat
        try:
            res.x
        except:
            self.fail_count += 1
            if self.fail_count <= 1:
                self.optimize(*args)

        return

    def in_space_placebo(self):
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
        placebo_outcomes = []
        for i in range(self.n_controls):
            #Format placebo and control data
            # treated_outcome: a (1 x treatment_period)
            treated_placebo_outcome = self.control_outcome_all[:,i].reshape(self.periods_all, 1)
            #treated_covariates: a (1 x len(covariates))
            treated_placebo_covariates = self.control_covariates[:,i].reshape(self.n_covariates, 1)
            print("treated")
            print(treated_placebo_outcome.shape, self.treated_outcome_all.shape)
            print(treated_placebo_covariates.shape,  self.treated_covariates.shape)
            # ((len(unit_list)-1) x treatment_period)
            control_placebo_outcome = np.array([self.control_outcome_all[:,j] for j in range(self.n_controls) if j != i]).T
            #(n_controls x len(covariates))
            control_placebo_covariates = np.array([[self.control_covariates[j,x] for x in range(self.n_controls) if x != i] for j in range(self.n_covariates)])
            print("controls")
            print(control_placebo_outcome.shape, self.control_outcome_all.shape)
            print(control_placebo_covariates.shape, self.control_covariates.shape)
            #Solve for best synthetic control weights
            self.optimize(treated_placebo_outcome[:self.treatment_period], 
                            treated_placebo_covariates,
                            control_placebo_outcome[:self.treatment_period], 
                            control_placebo_covariates,
                            True)
            
            #Compute outcome of best synthetic control
            #print(self.placebo_w.T)
            print(self.placebo_w.shape, treated_placebo_outcome.T.shape)
            synthetic_placebo_outcome = self.placebo_w @ treated_placebo_outcome.T
            #Store it
            placebo_outcomes.append(synthetic_placebo_outcome)
            print(placebo_outcomes)

        #Store numpy array
        self.in_space_placebos = placebo_outcomes
        return

    def in_time_placebo(self, treatment_period):
        '''
        Fits a synthetic control to the treated unit,
        at a treat

        Returns:
            matrix (n_controls x n_periods) with the outcome for each synthetic control
        '''
        return NotImplementedError

    def get_post_treatment_rmspe(self):
        '''
        Computes post-treatment outcome for treated and synthetic control unit
        Subtracts treatment effect from treated unit
        
        Returns post-treatment RMSPE for synthetic control.
        Required: self.w must be defined.
        '''
        
        #Get true counter factual by subtracting the treatment effect from the treated unit
        true_counterfactual = self.treated_outcome_all[self.treatment_period:] - self.treatment_effect
        synth = self.control_outcome_all[self.treatment_period:] @ self.w
        return np.sqrt(((true_counterfactual - synth) ** 2).mean())
    
    
    def get_pre_treatment_rmspe(self):
        '''
        Computes pre-treatment outcome for treated and synthetic control unit
        
        Returns pre-treatment RMSPE for synthetic control.
        Required: self.w must be defined.
        '''
        
        return np.sqrt(((self.treated_outcome - self.control_outcome @ self.w) ** 2).mean())
    
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
        
        if not include_donor_pool_average:
            
            table = pd.DataFrame({treated_label: self.treated_covariates.ravel(), 
                                 synth_label: (self.synth_covariates).ravel()},
                                index=self.covariates)
        
        else:
            
            table = pd.DataFrame({treated_label: self.treated_covariates.ravel(), 
                                 synth_label: (self.synth_covariates).ravel(),
                                 donor_pool_label: self.control_covariates.mean(axis=1)},
                                index=self.covariates)
        return table
    
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

    def add_constant(self):
        '''Method used only by Synthetic Diff-in-Diff'''
        constant = np.mean(self.treated_outcome - self.control_outcome @ self.w)
        self.control_outcome_all += constant