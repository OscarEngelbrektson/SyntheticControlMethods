import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.optimize import minimize, differential_evolution


class ControlSolver(object):
    
    def __init__(self, dataset, outcome_var, id_var, time_var, treatment_period, treated_unit, treatment_effect=None, w=None):
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
        
        treatment_effect:
        '''
        
        self.dataset = dataset
        self.y = outcome_var
        self.id = id_var
        self.time = time_var
        self.treatment_period = treatment_period
        self.treated_unit = treated_unit
        self.treatment_effect = treatment_effect #If known
        #All columns not y, id or time must be predictors
        self.covariates = [col for col in self.dataset.columns if col not in [self.y, self.id, self.time]]

        #Extract quantities needed for pre-processing matrices
        #Get number of periods in pre-treatment and total
        self.periods_pre_treatment = self.treatment_period - min(self.dataset[self.time])
        self.periods_all = max(self.dataset[self.time]) - min(self.dataset[self.time]) + 1
        #Number of control units, -1 to remove treated unit
        self.n_controls = len(self.dataset[self.id].unique()) - 1
        self.n_covariates = len(self.covariates)
        
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
        self.treated_outcome = None
        self.control_outcome = None
        self.treated_covariates = None
        self.control_covariates = None
        self.w = w #Can be provided if using Synthetic DID
        self.v = None
        
        
        self.treated_outcome_all = None
        self.control_outcome_all = None
        
        self.fail_count = 0 #Used to limit number of optimization attempts
    
    
    def transform_data(self):
        '''
        Takes an appropriately formatted, unprocessed dataset
        returns dataset with changes computed for the outcome variable
        Ready to fit a Difference-in-Differences Synthetic Control

        Transformation method - MeanSubtraction: 
        Subtracting the mean of the corresponding variable and unit from every observation
        '''
        mean_subtract_cols = self.dataset.groupby(self.id).apply(lambda x: x - np.mean(x)).drop(columns=[self.time], axis=1)
        return pd.concat([data[["ID", "Time"]], mean_subtract_cols], axis=1)

    
    def preprocess_data(self):
        '''
        Extracts processed variables from, excluding v and w, from input variables.
        These are all the data matrices.
        '''

        ###Get treated unit matrices first###
        treated_data_all = self.dataset[self.dataset[self.id] == self.treated_unit]
        #self.treated_outcome_all = np.array(treated_data_all[self.y]).T.reshape(1, self.periods_all).T #All outcomes
        self.treated_outcome_all = np.array(treated_data_all[self.y]).reshape(self.periods_all,1) #All outcomes
        
        #Only pre-treatment
        treated_data = treated_data_all[self.dataset[self.time] < self.treatment_period]
        #Extract outcome and shape as matrix
        self.treated_outcome = np.array(treated_data[self.y]).reshape(self.periods_pre_treatment, 1)
        #Columnwise mean of each covariate in pre-treatment period for treated unit, shape as matrix
        self.treated_covariates = np.array(treated_data[self.covariates].mean(axis=0)).reshape(self.n_covariates, 1)
        
        
        ### Now for control unit matrices ###
        #Every unit that is not the treated unit is control
        control_data_all = self.dataset[self.dataset[self.id] != self.treated_unit]
        self.control_outcome_all = np.array(control_data_all[self.y]).reshape(self.n_controls, self.periods_all).T #All outcomes
        
        #Only pre-treatment
        control_data = control_data_all[self.dataset[self.time] < self.treatment_period]
        #Extract outcome, then shape as matrix
        self.control_outcome = np.array(control_data[self.y]).reshape(self.n_controls, self.periods_pre_treatment).T
        
        #Extract the covariates for all the control units
        #Identify which rows correspond to which control unit by setting index, 
        #then take the unitwise mean of each covariate
        #This results in the desired (n_control x n_covariates) matrix
        self.control_covariates = np.array(control_data[self.covariates].\
                set_index(np.arange(len(control_data[self.covariates])) // self.periods_pre_treatment).\
                mean(level=0)).T
        
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
    
    def plot_outcome(self,
             normalized=False,
             title="Outcome of Treated Unit and Synthetic Control unit",
             treated_label="Treated unit",
             synth_label="Synthetic Control"):
        '''
        Plot the outcome of the Synthetic Unit against the Treated unit for both pre- and post-treatment periods
        '''
        
        #Extract Synthetic Control
        synth = self.w.T @ self.control_outcome_all.T #Transpose to make it (n_periods x 1)
        time = self.dataset[self.time].unique()
        
        plt.figure(figsize=(12, 8))
        plt.plot(time, synth.T, 'r--', label=synth_label)
        plt.plot(time ,self.treated_outcome_all, 'b-', label=treated_label)
        plt.title(title)
        #Mark where the last treatment period was, the last time we expect equal values
        plt.axvline(self.treatment_period-1, linestyle=':', color="gray")
        plt.annotate('Treatment', 
             xy=(self.treatment_period-1, self.treated_outcome[-1]*1.2),
             xytext=(-80, -4),
             xycoords='data',
             #textcoords="data",
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->"))
        plt.ylabel(self.y)
        plt.xlabel(self.time)
        plt.legend(loc='upper left')
        plt.show()
    
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
