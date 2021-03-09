from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

class SummaryTables(object):
    '''
    rmspe_df
    comparison_df
    weight_df
    '''
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
                                    "Importance":data.v,
                                    "Control Group Average": data.unscaled_control_covariates.mean(axis=1)},
                                    index=data.covariates)
      return comparison_df

    
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
            
            #Compute post/pre rmspe ratio and add to rmspe_df
            rmspe_df["post/pre"] = rmspe_df["post_rmspe"] / rmspe_df["pre_rmspe"]
            
            #Store it
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
