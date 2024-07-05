import os
import time
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.dates as mdates
import cftime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

from matplotlib.colors import BoundaryNorm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils.visualise import make_ailand_plot, plot_score_map
from utils.utils import r2_score_multi, anomaly_correlation, standardized_anomaly, get_scores_spatial_global, assemble_scores


class EvaluationModule:
    
    def __init__(self, forecast_module, lead_time, score = 'rmse', path_to_results = None):
        
        print("Model type is:", type(forecast_module.model))
        
        self.forecast_module = forecast_module
        self.init_score = score
        self.targ_idx = None
        self.path_to_results = path_to_results
        self.time_idxs = lead_time
        
        score_methods = {
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'acc': self.acc,
            'scaled_anom': self.scaled_anom
        }

        if score in score_methods:
            print("Evaluation with", score)
            self.score = score_methods[score]
        else:
            print("Don't know score!")
            
    def plot_heatmap(self, scores, times = None, discrete_classes = None, 
                     vmin = 0, threshold = None, cmap_style = 'OrRd', cbar_label = None,
                     style = 'rectangle', log=False, save_to = None, filename = ''):


        if style == 'pad_validtime':
            max_length = max(len(s) for s in scores)
            plot_scores = [np.pad(s, (max_length - len(s), 0), 'constant', constant_values=np.nan) for s in scores]
        elif style == 'pad_leadtime':
            max_length = max(len(s) for s in scores)
            plot_scores = [np.pad(s, (0, max_length - len(s)), 'constant', constant_values=np.nan) for s in scores]
        else:
            plot_scores = scores

        #truncated_scores_reversed = truncated_scores[::-1]
        if log:
            scores_matrix = np.log(np.vstack(plot_scores))
        else:
            scores_matrix = np.vstack(plot_scores)
            
        plt.figure(figsize=(10, 8))
        if discrete_classes is not None:
            cmap = plt.get_cmap(cmap_style)
            bounds = np.arange(0.1, 1.1, 0.1)  # Creates intervals from 0.1 to 1.0
            norm = BoundaryNorm(bounds, cmap.N)
            im = plt.imshow(scores_matrix.transpose(), aspect='auto', cmap=cmap, norm=norm, origin='lower')
            cbar = plt.colorbar(im, label='score', boundaries=bounds, ticks=bounds)
            im.set_clim(vmin=vmin, vmax=threshold) 
            tick_labels = [f"{x:.1f}" for x in bounds]
            cbar.set_ticks(bounds)
            cbar.set_ticklabels(tick_labels)
        else:
            im = plt.imshow(scores_matrix.transpose(), aspect='auto', cmap=cmap_style, origin='lower') 
            cbar = plt.colorbar(im, label=self.init_score)
            im.set_clim(vmin=vmin, vmax=threshold) 
            
        if cbar_label is None:
            cbar.set_label(label=f'{self.init_score}', fontsize=16)  
        else:
            cbar.set_label(label=cbar_label, fontsize=16)  
        #plt.title(f'Lead {self.score} by start times')
        if times is not None:
            times = np.array([np.datetime64(dt, 'D') for dt in times])
            times = np.array([dt.item().strftime('%Y-%m-%d') for dt in times])
            step_size = len(times) // 8
            tick_indices = np.arange(0, len(times), step_size)
            tick_indices = np.linspace(0, len(times) - 1, 6, dtype=int)
            tick_labels = times[tick_indices]
    
            plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)
            #plt.yticks(ticks=tick_indices, labels=tick_labels, rotation=45)

        plt.ylabel('Lead time step', fontsize=16)
        plt.xlabel('Initial forecast time', fontsize=16)
        plt.tight_layout()
        #plt.gca().invert_xaxis() 
        if save_to is not None:
            plt.savefig(os.path.join(save_to, f'fh_{self.init_score}_{filename}.pdf'))
        plt.show()

    def plot_timeseries(self, save_to):
        
        Y_prog, Y_prog_prediction = self.run_forecast(self.X_static, self.X_met, self.Y_prog)

        gc = np.random.choice(Y_prog.shape[1], 20)
        make_ailand_plot(Y_prog_prediction[:, gc, :], 
                     Y_prog[:, gc, :], 
                     Y_prog.shape[-1],
                    save_to = save_to)

    def plot_anomaly_persistence(self, preds_scores, ref_scores, persistence, save_to):

        plt.figure(figsize=(10, 8))
        plt.plot(persistence, color = "black")
        plt.plot(preds_scores, color="salmon", alpha = 0.8)
        plt.plot(ref_scores, color="blue", alpha = 0.8)
        plt.xlabel('Lead time', fontsize=16)
        plt.ylabel('Standardized anomaly', fontsize=16)
        plt.tight_layout()
        if save_to is not None:
            plt.savefig(save_to)
        plt.show()

    def plot_skillscore(self, skillscore, save_to):

        plt.figure(figsize=(10, 8))
        plt.hlines(y = 0, xmin = 0, xmax = self.time_idxs, color = 'black', linestyle = 'dashed')
        plt.plot(skillscore, color = "gray")
        plt.xlabel('Lead time', fontsize=16)
        plt.ylabel('Standardized anomaly', fontsize=16)
        plt.tight_layout()
        if save_to is not None:
            plt.savefig(save_to)
        plt.show()

    def rmse(self, x_preds, x_ref, **kwargs):
        return mean_squared_error(x_preds, x_ref, squared=False)

    def mae(self, x_preds, x_ref, **kwargs):
        return mean_absolute_error(x_preds, x_ref)

    def r2(self, x_preds, x_ref, **kwargs):
        return r2_score_multi(x_preds, x_ref)

    def acc(self, x_preds, x_ref, **kwargs):    
        return anomaly_correlation(x_preds, x_ref, kwargs["clim_mu"])

    def scaled_anom(self, x, **kwargs):    
        
        anom = standardized_anomaly(x, kwargs["clim_mu"], kwargs["clim_std"])
        anom = np.mean(anom)

        return anom

    def evaluate_total(self, Y_prog, Y_prog_prediction):
        eval_array = np.array([self.score(Y_prog[t, ...], 
                                          Y_prog_prediction[t, ...], 
                                          clim_mu = self.climatology_mu[t, ...],
                                         clim_std = self.climatology_std[t, ...]) for t in range(Y_prog_prediction.shape[0])])
        return eval_array

    def evaluate_target(self, Y_prog, Y_prog_prediction):
        eval_array = np.array([self.score(Y_prog[t, :, self.targ_idx, np.newaxis], 
                                          Y_prog_prediction[t, :, self.targ_idx, np.newaxis], 
                                          clim_mu = self.climatology_mu[t, :, self.targ_idx, np.newaxis],
                                         clim_std = self.climatology_std[t, :, self.targ_idx, np.newaxis]) for t in range(Y_prog_prediction.shape[0])])
        return eval_array

    def evaluate_skillscore_target(self, Y):
        eval_array = np.array([self.score(Y[t, :, self.targ_idx, np.newaxis],  
                                          clim_mu = self.climatology_mu[t, :, self.targ_idx, np.newaxis],
                                         clim_std = self.climatology_std[t, :, self.targ_idx, np.newaxis]) for t in range(Y.shape[0])])
        return eval_array

    def set_test_data(self, dataset):
        """
        Run outside of class.
        Args:
            dataset: specify from data_module.EcDataset.
        """
        # suboptimal to require forecast module here.
        self.targ_lst = dataset.targ_lst
        self.X_static, self.X_met, self.Y_prog = self.forecast_module.get_test_data(dataset)

    def set_climatology(self, file_path):

        """
        Needs to be called before computing skill scores.
        Evaluates to None if no path to climatology is provided in config.
        Args:
            file_path: Specify path to load climatology from.
            variability: Load climatological mean (False) or standard deviation (True)?
        """
        
        if file_path is not None:
            
            climatology = xr.open_dataset(file_path)
            climatology_mu = climatology['clim_6hr_mu'].sel(variable=self.targ_lst).values
            self.climatology_mu = np.tile(climatology_mu, (2, 1, 1))
            climatology_std = climatology['clim_6hr_std'].sel(variable=self.targ_lst).values
            self.climatology_std = np.tile(climatology_std, (2, 1, 1))

        else:
            self.climatology_mu = None
            self.climatology_std = None

    def set_target(self, target):
        """
        Needs to be specified before iterating initial times for target-specific horizons, otherwise will be None.
        """
        self.targ = target
        self.targ_idx = list(self.targ_lst).index(self.targ)

    def run_forecast(self, X_static, X_met, Y_prog):
        """
        Convenience function.
        """
        return self.forecast_module.step_forecast(X_static, X_met, Y_prog)
        
    def iterate_initial_times(self, start_idx):

        """
        Implemented for computing ACC forecast horizons.
        Args:
            start_idx: initial time to start forecast run.
        """
        
        print("Intital time at: ", start_idx)
        
        Y_prog_idx = self.Y_prog[start_idx:(start_idx + self.time_idxs), ...]
        X_met_idx = self.X_met[start_idx:(start_idx + self.time_idxs), ...]
        
        Y_prog_idx, Y_prog_prediction_idx = self.run_forecast(self.X_static, X_met_idx, Y_prog_idx)

        if self.targ_idx is None:
            print("Evaluating on all prognostic target states simulatenously")
            scores = self.evaluate_total(Y_prog_idx, Y_prog_prediction_idx)
        else:
            print("Evaluating on target: ", self.targ)
            scores = self.evaluate_target(Y_prog_idx, Y_prog_prediction_idx)

        return scores
        

    def iterate_initial_times_skillscore(self, start_idx):

        """
        Implemented for standardized anomaly persistence and for target-specific forecast horizons, i.e. not option of total evaluation. Creates a plot every 50th time step.
        Args:
            start_idx: initial time to start forecast run.
        """

        if self.targ_idx is None:
            print("SPECIFY TARGET!")
            
        print("Intital time at: ", start_idx)

        Y_prog_idx = self.Y_prog[start_idx:(start_idx + self.time_idxs), ...]
        X_met_idx = self.X_met[start_idx:(start_idx + self.time_idxs), ...]
        
        Y_prog_idx, Y_prog_prediction_idx = self.run_forecast(self.X_static, X_met_idx, Y_prog_idx)

        preds_scores = self.evaluate_skillscore_target(Y_prog_prediction_idx)
        ref_scores = self.evaluate_skillscore_target(Y_prog_idx)
        persistence = np.repeat(ref_scores[0], preds_scores.shape[0])
                        
        skillscore = 1 - np.abs(preds_scores[1:])/np.abs(persistence[1:])
        print(skillscore.shape)

        if start_idx % 50 == 0:
            print(f"saving figure on step {start_idx}...")   
            self.plot_anomaly_persistence(preds_scores, ref_scores, persistence, 
                                          save_to = os.path.join(self.path_to_results, f"anomaly_persistence_idx_{start_idx}"))
            self.plot_skillscore(skillscore, save_to = os.path.join(self.path_to_results, f"skillscore_idx_{start_idx}"))
        
        return skillscore