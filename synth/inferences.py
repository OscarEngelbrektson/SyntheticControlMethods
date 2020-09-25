from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import cvxpy as cvx
from scipy.optimize import minimize, differential_evolution

class Inferences(object):
    raise NotImplementedError

    def total_loss(self, v_0, optimize=True):
        '''
        Solves for w*(v) that minimizes loss function 1 given v,
        Returns loss from loss function 2 with w=w*(v)
        '''
        V = np.zeros(shape=(self.n_covariates, self.n_covariates))
        np.fill_diagonal(V, v_0)
        
        # Construct the problem - constrain weights to be non-negative
        w = cvx.Variable((self.n_controls, 1), nonneg=True)
        
        #Define the objective
        objective = cvx.Minimize(cvx.sum(V @ cvx.square(self.treated_covariates - self.control_covariates @ w)))
        
        #Add constraint sum of weights must equal one
        constraints = [cvx.sum(w) == 1]
        
        #Solve problem
        problem = cvx.Problem(objective, constraints)
        
        try: #Try solving using current value of V, if it doesn't work return infinite loss
            result = problem.solve(verbose=False)
            loss = (self.treated_outcome - self.control_outcome @ w.value).T @ (self.treated_outcome - self.control_outcome @ w.value)
        except:
            return float(np.inf)
       
        
        #If optimize is true, just return the loss
        if optimize:
            return loss
        
        else:
            #print("Total loss:", np.round(loss, 3))
            #print("Optimal w:", np.round(w.value, 3))
            return w.value, loss
    
    
    def optimize(self, steps=5, verbose=False):
        '''
        
        '''

        #Initalize variable to track best w*(v)
        best_w, min_loss = None, float(np.inf)
        
        for step in range(steps):

            #Dirichlet distribution returns a valid pmf over n_covariates states
            v_0 = np.random.dirichlet(np.ones(self.n_covariates), size=1)

            #Required to have non negative values
            bnds = tuple((0,1) for _ in range(self.n_covariates))

            #res = minimize(self.total_loss, v_0, bounds=bounds, method="L-BFGS-B")
            res = minimize(self.total_loss, v_0, method='L-BFGS-B', bounds=bnds, 
                          options={'disp':3, 'iprint':3}) #, constraints=cons) method='L-BFGS-B' tol=1e-20
            
            if verbose:
                print("Successful:", res.success)
                print(res.message)

            #If optimization was successful
            if res.success:
                #Compute w*(v) and loss for v
                w, loss = self.total_loss(res.x, False)

                #See if w*(v) results in lower loss, if so update best
                if loss < min_loss:
                    best_w, min_loss = w, loss
                    
                    #Store best v from optimization
                    self.v = res.x
        
        #If sampler did not converge, try again up to 3 times before admitting defeat
        try:
            best_w[0]
        except:
            self.fail_count += 1
            if self.fail_count <= 3:
                self.optimize()
            
        #Save best w
        self.w = best_w
        
        #Return total loss
        return min_loss
    
    
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
                                 synth_label: (self.control_covariates @ self.w).ravel()},
                                index=self.covariates)
        
        else:
            
            table = pd.DataFrame({treated_label: self.treated_covariates.ravel(), 
                                 synth_label: (self.control_covariates @ self.w).ravel(),
                                 donor_pool_label: self.control_covariates.mean(axis=1)},
                                index=self.covariates)
        return table