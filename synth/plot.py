#Will add plotting functionality here
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

class Plot(object):

    def plot(self, panels, figsize=(15, 12)):

        plt = self._get_plotter()
        fig = plt.figure(figsize=figsize)
        valid_panels = ['original']
        for panel in panels:
            if panel not in valid_panels:
                raise ValueError(
                    '"{}" is not a valid panel. Valid panels are: {}.'.format(
                        panel, ', '.join(['"{}"'.format(e) for e in valid_panels])
                    )
                )
        
        n_panels = len(panels)
        ax = plt.subplot(n_panels, 1, 1)
        idx = 1

        '''
         if 'original' in panels:
            ax.plot(pd.concat([self.pre_data.iloc[llb:, 0], self.post_data.iloc[:, 0]]),
                    'k', label='y')
            ax.plot(inferences['preds'], 'b--', label='Predicted')
            ax.axvline(inferences.index[intervention_idx - 1], c='k', linestyle='--')
            ax.fill_between(
                self.pre_data.index[llb:].union(self.post_data.index),
                inferences['preds_lower'],
                inferences['preds_upper'],
                facecolor='blue',
                interpolate=True,
                alpha=0.25
            )
            ax.grid(True, linestyle='--')
            ax.legend()
            if idx != n_panels:
                plt.setp(ax.get_xticklabels(), visible=False)
            idx += 1
        '''
        '''
        def plot_outcome(self,
             normalized=False,
             title="Outcome of Treated Unit and Synthetic Control unit",
             treated_label="Treated unit",
             synth_label="Synthetic Control"):
        '''
        #Plot the outcome of the Synthetic Unit against the Treated unit for both pre- and post-treatment periods
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
        '''

    def _get_plotter(self):  # pragma: no cover
        """Some environments do not have matplotlib. Importing the library through
        this method prevents import exceptions.

        Returns:
          plotter: `matplotlib.pyplot
        """
        import matplotlib.pyplot as plt
        return plt