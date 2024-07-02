import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.colors import LinearSegmentedColormap 
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import MaxNLocator

def plot_forecast_correlations(forecast1, forecast2, observation, variable, save_to=''):

    plt.figure(figsize=(9, 6))
    x = np.linspace(np.min(np.concatenate((forecast1[:, variable], forecast2[:, variable], observation[:, variable]))),
                    np.max(np.concatenate((forecast1[:, variable], forecast2[:, variable], observation[:, variable]))), 100)
    # Calculate the corresponding y-values (y = x)
    y = x
    # Create the scatterplot
    plt.plot(x, y, linestyle = '--', color = 'black')
    plt.scatter(forecast1[:, variable], observation[:, variable], alpha=0.6,color='blue', label='MLP')
    plt.scatter(forecast2[:, variable], observation[:, variable], alpha=0.7,color='salmon', label='LSTM')
    plt.xlabel("Forecast")
    plt.ylabel("Observation")
    plt.ylim((np.min(np.concatenate((forecast1[:, variable], forecast2[:, variable], observation[:, variable]))),
                     np.max(np.concatenate((forecast1[:, variable], forecast2[:, variable], observation[:, variable])))))
    plt.xlim((np.min(np.concatenate((forecast1[:, variable], forecast2[:, variable], observation[:, variable]))),
              np.max(np.concatenate((forecast1[:, variable], forecast2[:, variable], observation[:, variable])))))
    #plt.gca().set_aspect(x_range / y_range)
    #plt.ylim((-1.5,1.5))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

def plot_forecast_correlations_summarized(correlations1, correlations2, labs, save_to=''):

    plt.figure(figsize=(12, 6))
    plt.hlines(y=0,xmin=0, xmax=28, color='black', linestyle = '--', linewidth=0.8)
    plt.plot(np.arange(1,29), correlations1, marker='D', color='blue', label='MLP')
    plt.plot(np.arange(1,29), correlations2, marker='D', color='salmon', label='LSTM')
    plt.axvspan(11 - 0.5, 15 + 0.5, facecolor='lightgray', alpha=0.5)
    plt.axvspan(16 - 0.5, 20 + 0.5, facecolor='gray', alpha=0.5)
    plt.axvspan(21 - 0.5, 24 + 0.5, facecolor='lightgray', alpha=0.5)
    plt.axvspan(25 - 0.5, 28 + 0.5, facecolor='gray', alpha=0.5)
    plt.xticks(np.arange(1,29), labs, rotation= 80)
    plt.xlabel("Variable")
    plt.ylabel("Kendals Tau")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

def plot_skill_map(collected_skill, lead_time, variable=0, save_to=''):

    plt.figure(figsize=(7, 7))
    pos = np.arange(0, 72, (lead_time-24)/4)
    labs = ['Jan 2, 01:00', 'Jan 2, 18:00', 'Jan 3, 12:00', 'Jan 4, 06:00']
    plt.imshow(collected_skill[:,:,variable].transpose(), cmap='bwr_r', origin="lower", vmin=-1, vmax=1)
    plt.xticks(pos,labels=labs, rotation= 60)
    plt.ylabel("Lead time [hours]", fontweight= 'bold')
    plt.xlabel("Initial time", fontweight= 'bold')
    cb = plt.colorbar()
    cb.set_label('Skill')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

def plot_forecast_simple(y, forecast, variable, persistence= None, climatology = None, save_to = ''):

    plt.figure(figsize=(9, 6))
    plt.plot(y[:, variable], label= 'Observation')
    plt.plot(forecast[:, variable], label='Forecast')
    if persistence is not None:
        plt.plot(persistence[:, variable], label='Persistence')
    if climatology is not None:
        plt.plot(climatology[:, variable], label='Climatology')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel("Lead time ['hour']")
    plt.ylabel("Scaled target")
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

def plot_errors(forecast_errors, reference1_errors, reference2_errors, lead, error,
                variable = 0, save_to=''):
    plt.figure(figsize=(9, 6))
    plt.plot(np.arange(lead), forecast_errors[error][:, variable], linestyle='--', label='Forecast')
    plt.plot(np.arange(lead), reference1_errors[error][:, variable], linestyle='--', label='Persistence')
    plt.plot(np.arange(lead), reference2_errors[error][:, variable], linestyle='--', label='Climatology')
    plt.scatter(np.arange(lead), forecast_errors[error][:, variable], marker='D', label='Forecast')
    plt.scatter(np.arange(lead), reference1_errors[error][:, variable], marker='D', label='Persistence')
    plt.scatter(np.arange(lead), reference2_errors[error][:, variable], marker='D', label='Climatology')
    plt.xlabel("Lead time ['hour']")
    plt.ylabel("Absolute error")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

def plot_skill(skill1_mlp, skill2_mlp,
               lead, variable,skill1_lstm = None, skill2_lstm= None, save_to=''):

    plt.figure(figsize=(9, 5))
    plt.hlines(y=0, xmin=0, xmax=lead - 1, linewidth=0.8, colors='black')
    plt.plot(np.arange(lead), skill1_mlp[:, variable], linestyle='-',color = 'blue',label='MLP$_{Persistence}$')
    plt.plot(np.arange(lead), skill2_mlp[:, variable], linestyle='-', color = 'lightblue',label='MLP$_{Climatology}$')
    #plt.scatter(np.arange(lead), skill1_mlp[:, variable], marker='o', color = 'blue')
    #plt.scatter(np.arange(lead), skill2_mlp[:, variable], marker='o', color = 'lightblue')
    if not skill1_lstm is None:
        plt.plot(np.arange(lead), skill1_lstm[:, variable], linestyle='-',color = 'red', label='LSTM$_{Persistence}$')
        plt.plot(np.arange(lead), skill2_lstm[:, variable], linestyle='-', color = 'salmon', label='LSTM$_{Climatology}$')
        #plt.scatter(np.arange(lead), skill1_lstm[:, variable], marker='o', color = 'red')
        #plt.scatter(np.arange(lead), skill2_lstm[:, variable], marker='o',color = 'salmon', )
    plt.axhspan(ymin=np.concatenate((skill2_lstm[:, 0], skill2_mlp[:,0])).min(), ymax=0, facecolor='lightgray', alpha=0.3)
    plt.xlabel("Lead time ['hour']")
    plt.ylabel("Skill")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

def plot_losses(losses, which='loss', log=True, label = 'MSE', save_to=None):

    """
    :param losses: Takes the .csv output files from lightning logs.
    :param save_to: change direction if necessary
    :return: _
    """
    plt.figure(figsize=(8, 5)) 
    try:
        plot_train = losses[f'train_{which}'].to_numpy()
        plot_val = losses[f'val_{which}'].to_numpy()
    except KeyError as e:
        if str(e) == "'loss_logit'":
            plot_train = losses[f'train_{which}'].to_numpy()
            plot_val = losses[f'val_{which}'].to_numpy()
        else:
            plot_train = losses[f'train_loss'].to_numpy()
            plot_val = losses[f'val_loss'].to_numpy()
    if log:
        plot_train = np.log(plot_train)
        plot_val = np.log(plot_val)
            
    plt.plot(plot_train, marker='o', label='Training', color='blue')
    plt.plot(plot_val, marker='o', label='Validation', color='salmon')
    if log:
        plt.ylabel(f'Log({label})')
    else:
        plt.ylabel(f'{label}')
    plt.xlabel('Validation steps')
    plt.legend()
    plt.title(f'Losses: {which}')
    plt.grid(True)
    if save_to is not None:
        plt.savefig(os.path.join(save_to, f'losses_{which}.pdf'))
    plt.show()
    plt.close()

def plot_losses_combined(loss_files, titles, which='loss', log = True, label = 'MSE', save_to=None):
    """
    :param loss_files: A list of paths to .csv output files from lightning logs.
    :param titles: A list of titles for each subplot.
    :param which: Specifies the loss type to plot.
    :param save_to: Directory to save the figure if necessary.
    :return: _
    """
    num_files = len(loss_files)
    fig, axs = plt.subplots(1, num_files, figsize=(8 * num_files, 5))  # Adjusted figure size to accommodate multiple subplots
    
    if num_files == 1:  # Ensure axs is always iterable
        axs = [axs]
    
    for idx, loss_file in enumerate(loss_files):
        
        losses = pd.read_csv(loss_file)
        try:
            plot_train = losses[f'train_{which}'].to_numpy()
            plot_val = losses[f'val_{which}'].to_numpy()
        except KeyError as e:
            if str(e) == "'loss_logit'":
                plot_train = losses[f'train_{which}'].to_numpy()
                plot_val = losses[f'val_{which}'].to_numpy()
            else:
                plot_train = losses[f'train_loss'].to_numpy()
                plot_val = losses[f'val_loss'].to_numpy()
        if log:
            plot_train = np.log(plot_train)
            plot_val = np.log(plot_val)
            
        axs[idx].plot(plot_train, marker='o', label='Training', color='blue')
        axs[idx].plot(plot_val, marker='o', label='Validation', color='salmon')
        if log:
            axs[idx].set_ylabel(f'Log({label})')
        else:
            axs[idx].set_ylabel(f'{label}')
        axs[idx].set_xlabel('Epoch')
        axs[idx].legend()
        axs[idx].set_title(f'Logit loss: {titles[idx]}')
        axs[idx].grid(True)
    
    plt.tight_layout()  # Adjust layout to not overlap
    
    if save_to is not None:
        plt.savefig(os.path.join(save_to, f'combined_losses_{which}.pdf'))
    
def plot_losses_targetwise(losses, which = "loss", log = True, label = 'MSE', targ_lst = ['swvl1',
                         'swvl2',
                         'swvl3',
                         'stl1',
                         'stl2',
                         'stl3',
                         'snowc',
                         ], save_to=None):


    targ_lst = ["Soil Moisture Layer 1", "Soil Moisture Layer 2", "Soil Moisture Layer 3", 
                "Soil Temperature Layer 1", "Soil Temperature Layer 2", "Soil Temperature Layer 3", 
                "Snow Cover Fraction"]
    
    num_variables = len(targ_lst)
    num_columns = 3
    num_rows = (num_variables + num_columns - 1) // num_columns  # Ceiling division to ensure enough rows

    # Create subfigures with adjusted layout
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 3))  # Adjusted figsize as needed

    # Flatten axs for easy indexing
    axs = axs.flatten()

    for i in range(num_variables):
        column_name1 = f'train_{which}_var_{i}'
        column_name2 = f'val_{which}_var_{i}'
        if log:
            axs[i].plot(np.log(losses[column_name1].dropna()), label='Training')
            axs[i].plot(np.log(losses[column_name2].dropna()), label='Validation')
            axs[i].set_ylabel(f'Log({label})')
        else:
            axs[i].plot(losses[column_name1].dropna(), label='Training')
            axs[i].plot(losses[column_name2].dropna(), label='Validation')
            axs[i].set_ylabel(f'{label}')
        axs[i].set_title(targ_lst[i])
        axs[i].set_xlabel('Validation steps')
        axs[i].legend()

    # Remove empty subplots
    for j in range(num_variables, num_rows * num_columns):
        fig.delaxes(axs[j])

    plt.tight_layout()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, f'{which}_targetwise.pdf'))
    plt.show()
    
def plot_losses_targetwise_boxplots(losses, which="loss", log=True, label='SmoothL1', targ_lst=None, save_to=None):
    
    if targ_lst is None:
        targ_lst = ['swvl1', 'swvl2', 'swvl3', 'stl1', 'stl2', 'stl3', 'snowc']
        
    targ_lst_labels = ["Soil Moisture Layer 1", "Soil Moisture Layer 2", "Soil Moisture Layer 3", 
                       "Soil Temperature Layer 1", "Soil Temperature Layer 2", "Soil Temperature Layer 3", 
                       "Snow Cover Fraction"]
    
    # Create subplots with adjusted layout
    fig, axs = plt.subplots(1, 1, figsize=(9, 9))  # Adjusted figsize as needed

    val_losses = []
    for i in range(len(targ_lst)):
        column_name = f'val_{which}_var_{i}'
        if column_name in losses.columns:
            val_losses.append(losses[column_name].dropna())
        else:
            print(f"Warning: {column_name} not found in dataframe columns.")
            val_losses.append(pd.Series(dtype=float))

    if log:
        box = axs.boxplot([np.log(loss) for loss in val_losses if not loss.empty], vert=True, patch_artist=True)
        axs.set_ylabel(f'Log({label})', fontsize=16)
    else:
        box = axs.boxplot([loss for loss in val_losses if not loss.empty], vert=True, patch_artist=True)
        axs.set_ylabel(f'{label}', fontsize=16)


    axs.set_xticklabels(targ_lst_labels, rotation=45, ha='right')

    axs.tick_params(axis='x', labelsize=16)
    axs.tick_params(axis='y', labelsize=16)

    for box_element in ['boxes', 'whiskers', 'caps', 'medians', 'means']:
        plt.setp(box[box_element], color='black', linewidth=2)
    for patch in box['boxes']:
        patch.set_facecolor('gray')
        
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, f'{which}_bp_targetwise.pdf'))
    plt.show()

def plot_cumulative_performances(collected_performance1, collected_performance2, rollouts, variable = 0, save_to=''):

    plt.plot(rollouts, collected_performance1[:,variable], linestyle='--', color = 'blue')
    plt.plot(rollouts, collected_performance2[:,variable], linestyle='--', color = 'red')
    plt.scatter(rollouts, collected_performance1[:,variable], marker='D', color = 'blue', label='MLP')
    plt.scatter(rollouts, collected_performance2[:,variable], marker='D', color = 'red', label='LSTM')
    plt.xticks(np.arange(1,13), labels=np.arange(1,13))
    plt.xlabel("Trained at lead time ['hour']")
    plt.ylabel("MAE")
    plt.ylim((0,0.125))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def boxplot_summarized_performances(collected_performances1, collected_performances2, save_to=''):
    labs = ['AvgSurfT', 'CanopInt', 'HLICE', 'SAlbedo', 'SWE', 'TLBOT',
           'TLICE', 'TLMNW', 'TLSF', 'TLWML', 'SWEML_0', 'SWEML_1', 'SWEML_2',
           'SWEML_3', 'SWEML_4', 'SnowTML_0', 'SnowTML_1', 'SnowTML_2',
           'SnowTML_3', 'SnowTML_4', 'SoilTemp_0', 'SoilTemp_1', 'SoilTemp_2',
           'SoilTemp_3', 'SoilMoist_0', 'SoilMoist_1', 'SoilMoist_2',
           'SoilMoist_3']
    pos = np.arange(1,29)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(2,1, figsize=(12, 8))
    boxprops = dict(linewidth=2, color='blue')
    medianprops = dict(linewidth=2, color='red')
    axs[0].axvspan(11-0.5, 15+0.5, facecolor='lightgray', alpha=0.5)
    axs[0].axvspan(16-0.5, 20+0.5, facecolor='gray', alpha=0.5)
    axs[0].axvspan(21-0.5, 24+0.5, facecolor='lightgray', alpha=0.5)
    axs[0].axvspan(25-0.5, 28+0.5, facecolor='gray', alpha=0.5)
    axs[0].axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    axs[0].axhline(y=np.log(0.1), color='black', linewidth=0.8, linestyle='--')
    axs[0].boxplot(np.log(collected_performances1), vert=True, boxprops=boxprops, medianprops=medianprops)
    axs[0].set_title('MLP')
    axs[0].set_xticklabels([])  # Remove y-axis labels
    axs[0].grid(axis='x', linestyle='--', alpha=0.7)
    axs[0].set_ylim((-5,1.5))
    axs[0].set_ylabel('log(MAE)')
    axs[1].axvspan(11-0.5, 15+0.5, facecolor='lightgray', alpha=0.5)
    axs[1].axvspan(16-0.5, 20+0.5, facecolor='gray', alpha=0.5)
    axs[1].axvspan(21-0.5, 24+0.5, facecolor='lightgray', alpha=0.5)
    axs[1].axvspan(25-0.5, 28+0.5, facecolor='gray', alpha=0.5)
    axs[1].axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    axs[1].axhline(y=np.log(0.1), color='black', linewidth=0.8, linestyle='--')
    axs[1].boxplot(np.log(collected_performances2), vert=True, boxprops=boxprops, medianprops=medianprops)
    axs[1].set_title('LSTM')
    axs[1].grid(axis='x', linestyle='--', alpha=0.7)
    axs[1].set_xticks(pos, labs, rotation= 80)
    axs[1].set_ylim((-5,1.5))
    axs[1].set_ylabel('log(MAE)')
    plt.tight_layout()
    plt.savefig(save_to)


def plot_forecast(yhat, yosm, all=False,
                  rmse = None, skill = None, persistence = None, yhat2=None,
                  model = ['MLP'], save_to='figures'):

    if all:
        plt.rcParams.update({'font.size': 16})
        labs = ['AvgSurfT', 'CanopInt', 'HLICE', 'SAlbedo', 'SWE', 'TLBOT',
       'TLICE', 'TLMNW', 'TLSF', 'TLWML', 'SWEML_0', 'SWEML_1', 'SWEML_2',
       'SWEML_3', 'SWEML_4', 'SnowTML_0', 'SnowTML_1', 'SnowTML_2',
       'SnowTML_3', 'SnowTML_4', 'SoilTemp_0', 'SoilTemp_1', 'SoilTemp_2',
       'SoilTemp_3', 'SoilMoist_0', 'SoilMoist_1', 'SoilMoist_2',
       'SoilMoist_3']
        nrows = 6
        ncols = 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(16, 14), gridspec_kw={'hspace': 0.99, 'wspace': 0.3}, sharex=False)
        fig.text(2, 1, 'Timestep [hour]', ha='center')
        count = 0
        for i in range(nrows):
            for j in range(ncols):
                if count == 28:
                    break
                else:
                    if not persistence is None:
                        axs[i, j].plot(persistence[:, count], color='#386cb0', alpha=0.8, linestyle='--', label = 'Persistence')
                    if not yhat2 is None:
                        axs[i, j].plot(yhat2[:, count], color='#ca0020', alpha=0.8, label = model[1])
                    axs[i, j].plot(yosm[:, count], color='black', alpha=0.8, label='OSM')
                    axs[i, j].plot(yhat[:, count], color='#f4a582', alpha=0.8, label = model[0])
                    if not rmse is None:
                        axs[i, j].set_title(f'{labs[count]} \n RMSE: {np.format_float_positional(rmse[count], 3)}')
                    if not skill is None:
                        axs[i, j].set_title(f'{labs[count]} \n Skill: {np.format_float_positional(skill[count], 3)}')
                    else:
                        axs[i, j].set_title(f'{labs[count]}')
                    count += 1
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        fig.text(0.5, 0.04, 'Hour since $t=0$', ha='center')
        #fig.tight_layout()
        fig.savefig(save_to)
    else:
        labs = ['AvgSurfT', 'SoilMoist_0']
        plt.rcParams.update({'font.size': 11})
        skill = np.abs(np.subtract(yhat, yosm)) - np.abs(np.subtract(persistence, yosm))
        if not yhat2 is None:
            skill2 = np.abs(np.subtract(yhat2, yosm)) - np.abs(np.subtract(persistence, yosm))
        fig, axs = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [3, 1.5, 3, 1.5]}, figsize=(5, 7))
        x = np.arange(1, len(yhat))
        axs[0].plot(yosm[:, 0], color='#1f78b4', label='OSM', alpha=0.8)
        axs[0].plot(yhat[:, 0], color='#f4a582', label=model[0], alpha=0.7)
        axs[0].plot(x, persistence[x, 0], color='black', linestyle='--', label='Persistence', alpha=0.3)
        if not yhat2 is None:
            axs[0].plot(yhat2[:, 0], color='#ca0020', alpha=0.8, label=model[1])
        axs[0].legend(loc='lower right')
        axs[0].set_ylabel('AvgSurfT [K]')
        axs[1].axhline(y=0, color='lightgray', linewidth=0.8)
        if not yhat2 is None:
            axs[1].plot(skill[:, 0], color='#f4a582', linewidth=0.9)
            axs[1].plot(skill2[:, 0], color='#ca0020', linewidth=0.9)
        else:
            axs[1].plot(skill[:, 0], color='black', linewidth=0.9)
        axs[1].set_ylabel('Skill')
        axs[1].axhspan(ymin=-2, ymax=0, facecolor='lightgray', alpha=0.3)
        #axs[1].set_ylim((-max(abs(skill[:, 0])), max(abs(skill[:, 0]))))
        axs[2].plot(yosm[:, 1], color='#1f78b4', label='OSM', alpha=0.8)
        axs[2].plot(yhat[:, 1], color='#f4a582', label=model[0], alpha=0.7)
        axs[2].plot(x, persistence[x, 1], color='black', linestyle='--', label='Persistence', alpha=0.3)
        if not yhat2 is None:
            axs[2].plot(yhat2[:, 1], color='#ca0020', alpha=0.8, label=model[1])
        axs[2].set_ylabel('SoilMoist0 [kg m-2]')
        #axs[2].legend()
        axs[3].axhline(y=0, color='lightgray', linewidth=0.8)
        if not yhat2 is None:
            axs[3].plot(skill[:,1 ], color='#f4a582', linewidth=0.9)
            axs[3].plot(skill2[:, 1], color='#ca0020', linewidth=0.9)
        else:
            axs[3].plot(skill[:, 1], color='black', linewidth=0.9)
        axs[3].axhspan(ymin=-2, ymax=0, facecolor='lightgray', alpha=0.3)
        axs[3].set_ylabel('Skill')
        #axs[3].set_ylim((-max(abs(skill[:, ])), max(abs(skill[:, ]))))
        axs[3].set_xlabel('Timesteps [hours]')
        plt.tight_layout()
        plt.savefig(save_to)
        plt.close()


def plot_estimations(yosm, yhat, save_to='figures',
                     y1label = 'GPP [mg C m-2 s-1]', y2label = 'EVAP [mg m-2 s-1]',
                     legend_label = '$\hat{y}_{mlp_1D}$'):

    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [3, 1, 3, 1]}, figsize=(6, 8))
    axs[0].plot(yosm[:, 0], color="blue", alpha=0.8, linewidth=0.8, label='$\hat{y}_{osm}$')
    axs[0].plot(yhat[:, 0], color="salmon", alpha=0.7, linewidth=0.8, label=legend_label)
    axs[0].set_ylabel(y1label)
    axs[0].get_yaxis().set_label_coords(-0.1, 0.5)
    axs[0].legend()
    axs[1].plot(np.subtract(yhat[:, 0], yosm[:, 0]), color="gray", linewidth=0.8)
    axs[1].set_ylabel('Bias')
    axs[1].get_yaxis().set_label_coords(-0.1, 0.5)
    axs[3].set_ylim((-2.5, 2.5))
    axs[2].plot(yosm[:, 1], color="blue", alpha=0.8, linewidth=0.8, label='$\hat{y}_{osm}$')
    axs[2].plot(yhat[:, 1], color="salmon", alpha=0.7, linewidth=0.8, label=legend_label)
    axs[2].set_ylabel(y2label)
    axs[2].get_yaxis().set_label_coords(-0.1, 0.5)
    axs[3].plot(np.subtract(yhat[:, 1], yosm[:, 1]), color="gray", linewidth=0.8)
    axs[3].set_ylabel('Bias')
    axs[3].get_yaxis().set_label_coords(-0.1, 0.5)
    axs[3].set_ylim((-5,5))
    plt.xlabel('Timestep [hour]')
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(save_to)
    plt.close()

def plot_forecast_lead(forecast1, forecast2, yosm, reference1, reference2, eval1, eval2,
                       variable = 0, save_to=''):
    plt.rcParams.update({'font.size': 18})
    labs = ['AvgSurfT', 'CanopInt', 'HLICE', 'SAlbedo', 'SWE', 'TLBOT',
            'TLICE', 'TLMNW', 'TLSF', 'TLWML', 'SWEML_0', 'SWEML_1', 'SWEML_2',
            'SWEML_3', 'SWEML_4', 'SnowTML_0', 'SnowTML_1', 'SnowTML_2',
            'SnowTML_3', 'SnowTML_4', 'SoilTemp_0', 'SoilTemp_1', 'SoilTemp_2',
            'SoilTemp_3', 'SoilMoist_0', 'SoilMoist_1', 'SoilMoist_2',
            'SoilMoist_3']
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1.5]},
                            figsize=(9, 6))
    #x = np.arange(lead, len(yhat) + lead)
    axs[0].plot(yosm[:, variable], color='black', label='ECLand')
    axs[0].plot(forecast1[:, variable], color='blue', label='MLP')
    axs[0].plot(forecast2[:, variable], color='salmon', label='LSTM')
    axs[0].plot(reference1[:, variable], color='lightgray', linestyle='--', label='Persistence')
    axs[0].plot(reference2[:, variable], color='gray', linestyle='--', label='Climatology')
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[0].set_ylabel(labs[variable])
    axs[1].axhline(y=0, color='black', linewidth=0.8)
    axs[1].plot(eval1[:, variable], color='blue', linewidth=1.0)
    axs[1].plot(eval2[:, variable], color='salmon', linewidth=1.0)
    # axs[1].plot(x, np.abs(np.subtract(persistence[:, 0], yosm[:,0])), color='darkgray', linestyle='--', label='Persistance')
    axs[1].set_ylabel('Residuals')
    axs[1].axhspan(ymin=-2, ymax=0, facecolor='white', alpha=0.3)
    axs[1].set_ylim((np.min(np.concatenate((eval1[:, variable], eval2[:, variable]))), np.max(np.concatenate((eval1[:, variable], eval2[:, variable])))))
    axs[1].set_xlabel('Lead time [hours]')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()

    
    
def make_ailand_plot(Preds, X_val, target_size, save_to=None):
    
    fig = plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(331)
    ax2 = plt.subplot(332)
    ax3 = plt.subplot(333)
    ax4 = plt.subplot(334)
    ax5 = plt.subplot(335)
    ax6 = plt.subplot(336)
    ax7 = plt.subplot(337)

    def ailand_plot(idx, ax, ylabel, ax_title, test_date="2022-01-01"):
        """Plotting function for the ec-land database and ai-land model output

        :param var_name: parameter variable name
        :param ax: the axes to plot on
        :param ylabel: y-label for plot
        :param ax_title: title for plot
        :param test_date: date to plot vertical line (train/test split), defaults to "2022-01-01"
        :return: plot axes
        """
        # ax.plot(feats_arr_rf[:, idx], label="rf-land")
        ax.plot(Preds[:, :, -target_size + idx], label="ai-land", color="salmon")
        ax.plot(X_val[:, :, -target_size + idx], label="ec-land", color="blue", alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(ax_title)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # This removes duplicates.
        ax.legend(by_label.values(), by_label.keys())
        return ax

    ailand_plot(0, ax1, "Soil Moisture (m3 m-3)", "Soil Moisture Layer 1")
    ailand_plot(1, ax2, "Soil Moisture (m3 m-3)", "Soil Moisture Layer 2")
    ailand_plot(2, ax3, "Soil Moisture (m3 m-3)", "Soil Moisture Layer 3")
    ailand_plot(3, ax4, "Soil Temperature (K)", "Soil Temperature Layer 1")
    ailand_plot(4, ax5, "Soil Temperature (K)", "Soil Temperature Layer 2")
    ailand_plot(5, ax6, "Soil Temperature (K)", "Soil Temperature Layer 3")
    ailand_plot(6, ax7, "Snow Cover Fraction (-)", "Snow Cover Fraction")

    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)

def plot_score_map(performances, error='r2', vmin=None, vmax=None, cmap="BrBG", 
                   transparent=False, save_to=None, file='', ax=None, cb = None):
    if ax is None:
        # Create a new figure and axes with cartopy projection if not provided
        fig, ax = plt.subplots(figsize=(11, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        own_figure = True
    else:
        # Use the provided axes and assume the calling code handles figure creation
        own_figure = False
    
    # Remove frame spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if own_figure:
        fig.patch.set_visible(False)
    ax.axis('off')
    
    # Adding gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, 
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 15, 'color': 'lightgray'}
    gl.ylabel_style = {'size': 15, 'color': 'lightgray'}

    ax.coastlines()

    # Scatter plot
    scatter = ax.scatter(performances['lon'], performances['lat'], c=performances[error],
                         cmap=cmap, s=36, alpha=0.75, vmin=vmin, vmax=vmax, 
                         edgecolor=None, transform=ccrs.PlateCarree())
    if cb is not None:
        cbar = plt.colorbar(scatter, ax=ax, extend='neither', label=error)
        cbar.ax.set_ylabel(error, labelpad=15, fontsize=15)
        cbar.ax.tick_params(labelsize=15)

    if save_to is not None:
        plt.savefig(os.path.join(save_to, f'score_map_{error}_{file}.pdf'), transparent=transparent)
        plt.show()

            
def plot_map(lat, lon, var, vmin=None, vmax=None, cmap="BrBG", s=35,label = None,
                   transparent=False, save_to=False, file='', ax=None):
    if ax is None:
        # Create a new figure and axes with cartopy projection if not provided
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        own_figure = True
    else:
        # Use the provided axes and assume the calling code handles figure creation
        own_figure = False
    
    # Remove frame spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if own_figure:
        fig.patch.set_visible(False)
    ax.axis('off')
    
    # Adding gridlines
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, 
    #                  color='gray', alpha=0.5, linestyle='--')
    #gl.top_labels = False
    #gl.right_labels = False
    #gl.xlabel_style = {'size': 15, 'color': 'lightgray'}
    #gl.ylabel_style = {'size': 15, 'color': 'lightgray'}

    ax.coastlines()

    # Scatter plot
    scatter = ax.scatter(lat, lon, c=var,
                         cmap=cmap, s=s, alpha=0.8, vmin=vmin, vmax=vmax, 
                         edgecolor=None, transform=ccrs.PlateCarree())

    # Adding colorbar
    if own_figure:
        cbar = plt.colorbar(scatter, ax=ax, extend='neither')
        if label is not None:
            cbar.ax.set_ylabel(label, labelpad=15, fontsize=16)
        cbar.ax.tick_params(labelsize=15)

    if save_to and own_figure:
        plt.savefig(os.path.join(save_to, f'map_{file}.pdf'), transparent=transparent)
        plt.show()
        if own_figure:
            plt.close(fig)


def plot_correlation_scatter(y, y_hat, target, save_to=None):

    y = y.flatten()
    y_hat = y_hat.flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y, y_hat, alpha=0.5)

    # Plot the correlation line with a slope of 1
    min_val = min(min(y), min(y_hat))
    max_val = max(max(y), max(y_hat))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    # Scale the x and y axes to the same range
    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])

    # Add labels and title
    plt.xlabel(f'{target} ECland', fontsize=14)
    plt.ylabel(f'{target} emulator', fontsize=14)
    plt.legend()


    # Save plot to file if specified
    if save_to is not None:
        plt.savefig(save_to)

    # Show plot
    plt.show()


def plot_correlation_scatter_binned(y, y_hat, target, save_to=None):

    bin_centers = []
    means = []
    std_devs = []

    for i in range(len(y)):
        y_i = y[i].flatten()
        y_hat_i = y_hat[i].flatten()
    
        # Define quantile for binning
        quantiles = np.linspace(0, 1, 11)
    
        # Compute quantile bins
        bin_edges = np.quantile(y_i, quantiles)
        
        bin_centers_i = []
        means_i = []
        std_devs_i = []
        
        for i in range(len(bin_edges) - 1):
            bin_mask = (y_i >= bin_edges[i]) & (y_i < bin_edges[i + 1])
            y_bin = y_i[bin_mask]
            y_hat_bin = y_hat_i[bin_mask]
            
            if len(y_bin) > 0:
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                bin_centers_i.append(bin_center)
                means_i.append(y_hat_bin.mean())
                std_devs_i.append(y_hat_bin.std())

        bin_centers.append(bin_centers_i)
        means.append(means_i)
        std_devs.append(std_devs_i)
        
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the correlation line with a slope of 1
    min_val = min(min(y_i.flatten()), min(y_hat_i.flatten()))
    max_val = max(max(y_i.flatten()), max(y_hat_i.flatten()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=1, color="black")

    colors = ['lightgreen', 'goldenrod', 'blue']
    labels = ['XGB', 'MLP', 'LSTM']
    for i in range(len(bin_centers)):
        ax.errorbar(bin_centers[i], means[i], yerr=std_devs[i], color=colors[i], fmt='o',markersize=12, alpha=0.9)
        ax.plot(bin_centers[i], means[i], alpha=0.9, color=colors[i],linewidth=2, label = labels[i])

    # Scale the x and y axes to the same range
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    # Add labels and title
    ax.set_xlabel(f'ECland prediction', fontsize=20)
    ax.set_ylabel(f'Emulator prediction', fontsize=20)
    ax.set_title(f'{target}', fontsize=20)
    ax.legend()

    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)

    # Show plot
    plt.show()
    plt.close()

# Example usage
# y = np.random.rand(1000)
# y_hat = y + np.random.normal(0, 0.1, 1000)
# plot_correlation_scatter(y, y_hat, target='Example Target')

            
def boxplot_scores_total(xgb_scores, mlp_scores, lstm_scores, log = False, data='europe', target = '', save_to = ''):
    
    if log:
        data_rmse = [ np.log(xgb_scores['rmse']),  np.log(mlp_scores['rmse']), np.log(lstm_scores['rmse'])]
        data_mae = [np.log(xgb_scores['mae']), np.log(mlp_scores['mae']), np.log(lstm_scores['mae'])]
        data_r2 = [xgb_scores['r2'],  mlp_scores['r2'],  lstm_scores['r2']]
        #data_mape = [np.log(xgb_scores['mape']), np.log(mlp_scores['mape']), np.log(lstm_scores['mape'])]
    else:
        data_rmse = [ xgb_scores['rmse'],  mlp_scores['rmse'], lstm_scores['rmse']]
        data_mae = [xgb_scores['mae'], mlp_scores['mae'], lstm_scores['mae']]
        data_r2 = [xgb_scores['r2'],  mlp_scores['r2'],  lstm_scores['r2']]
        #data_mape = [xgb_scores['mape'],mlp_scores['mape'],lstm_scores['mape']]
        
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    colors = ['lightgreen', 'yellow', 'blue']
    median_color = 'black'
    
    bp_rmse = axs[0].boxplot(data_rmse, patch_artist=True)
    axs[0].set_xticks([1, 2, 3])
    axs[0].set_xticklabels(["XGB", "MLP", "LSTM"], fontsize=16)
    axs[0].set_ylabel('Log(RMSE)', fontsize=16)
    #axs[0,0].set_title(f'Root Mean Squared Error: {period}')
    for patch, color in zip(bp_rmse['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp_rmse['medians']:
        median.set_color(median_color)
    
    bp_mae = axs[1].boxplot(data_mae, patch_artist=True)
    axs[1].set_xticks([1, 2, 3])
    axs[1].set_xticklabels(["XGB", "MLP", "LSTM"], fontsize=16)
    axs[1].set_ylabel('Log(MAE)', fontsize=16)
    #axs[0,1].set_title(f'Mean Absolute Error: {period}')
    for patch, color in zip(bp_mae['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp_mae['medians']:
        median.set_color(median_color)
    
    bp_r2 = axs[2].boxplot(data_r2, patch_artist=True)
    axs[2].set_xticks([1, 2, 3])
    axs[2].set_xticklabels(["XGB", "MLP", "LSTM"], fontsize=16)
    axs[2].set_ylabel('R2', fontsize=16)
    axs[2].set_ylim((0.0,1.03))
    #axs[1,0].set_title(f'R2 Score: {period}, truncated at 0')
    for patch, color in zip(bp_r2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp_r2['medians']:
        median.set_color(median_color)
       
    plt.tight_layout()
    plt.savefig(os.path.join(save_to, f'boxplot_scores_total_{data}_{target}.pdf'))
    plt.show()
    
def boxplot_scores_single(xgb_scores, mlp_scores, lstm_scores, score = 'mae', log = False, data='europe', target = '', save_to = ''):
    
    if log:
        scores = [ np.log(xgb_scores[score]),  np.log(mlp_scores[score]), np.log(lstm_scores[score])]
    else:
        scores = [ xgb_scores[score],  mlp_scores[score], lstm_scores[score]]
        
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    
    colors = ['lightgreen', 'yellow', 'blue']
    median_color = 'black'
    
    bp_rmse = axs.boxplot(scores, patch_artist=True)
    axs.set_xticks([1, 2, 3])
    axs.set_xticklabels(["XGB", "MLP", "LSTM"], fontsize=16)
    if log:
        axs.set_ylabel(f"Log({score})", fontsize=16)
    else:
        axs.set_ylabel(f"{score}", fontsize=16)
    #axs[0,0].set_title(f'Root Mean Squared Error: {period}')
    for patch, color in zip(bp_rmse['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp_rmse['medians']:
        median.set_color(median_color)
       
    plt.tight_layout()
    plt.savefig(os.path.join(save_to, f'boxplot_scores_single_{data}_{score}_{target}.pdf'))
    plt.show()
    plt.close()

def boxplot_scores_single_new(xgb_scores, mlp_scores, lstm_scores, score='mae', log=False, data='europe', target='', save_to=''):
    scores = [xgb_scores[score], mlp_scores[score], lstm_scores[score]]
        
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    
    colors = ['lightgreen', 'yellow', 'blue']
    median_color = 'black'
    
    bp_rmse = axs.boxplot(scores, patch_artist=True)
    axs.set_xticks([1, 2, 3])
    axs.set_xticklabels(["XGB", "MLP", "LSTM"], fontsize=16)
    axs.set_ylabel(f"{score}", fontsize=16)
    
    if log:
        axs.set_yscale('log')
    
    for patch, color in zip(bp_rmse['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp_rmse['medians']:
        median.set_color(median_color)
       
    plt.tight_layout()
    plt.savefig(os.path.join(save_to, f'boxplot_scores_single_{data}_{score}_{target}.pdf'))
    plt.show()
    plt.close()

def boxplot_scores_leadtimes(leadtime_scores, log = False, data='europe', target = None, save_to = ''):
    
    model_scores = list(leadtime_scores.values())
    if target is None:
        target = 'total'
        print("Target:", target)
        if log:
            data_rmse = [ np.log(model['rmse']) for model in model_scores]
            data_mae = [ np.log(model['mae']) for model in model_scores]
            data_r2 = [ model['r2'] for model in model_scores]
        else:
            data_rmse = [ model['rmse'] for model in model_scores]
            data_mae = [ model['mae'] for model in model_scores]
            data_r2 = [ model['r2'] for model in model_scores]
    else:
        print("Target:", target)
        if log:
            data_rmse = [ np.log(model[target]['rmse']) for model in model_scores]
            data_mae = [ np.log(model[target]['mae']) for model in model_scores]
            data_r2 = [ model[target]['r2'] for model in model_scores]
        else:
            data_rmse = [ model[target]['rmse'] for model in model_scores]
            data_mae = [ model[target]['mae'] for model in model_scores]
            data_r2 = [ model[target]['r2'] for model in model_scores]
    
    labels = [f"Lookback {i*10}" for i in range(1,len(model_scores)+1)]
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    
    cmap = LinearSegmentedColormap.from_list("blue_grad", ["lightblue", "darkblue"], N=len(labels))
    median_color = 'black'
    
    bp_rmse = axs[0].boxplot(data_rmse, patch_artist=True)
    axs[0].set_xticks(list(np.arange(1,len(labels)+1)))
    axs[0].set_xticklabels(labels, fontsize=16, rotation = 45)
    axs[0].set_ylabel('Log(RMSE)', fontsize=16)
    #axs[0,0].set_title(f'Root Mean Squared Error: {period}')
    for patch, color in zip(bp_rmse['boxes'], np.arange(len(labels))):
        patch.set_facecolor(cmap(color /len(labels)))
    for median in bp_rmse['medians']:
        median.set_color(median_color)
    
    bp_mae = axs[1].boxplot(data_mae, patch_artist=True)
    axs[1].set_xticks(list(np.arange(1,len(labels)+1)))
    axs[1].set_xticklabels(labels, fontsize=16, rotation = 45)
    axs[1].set_ylabel('Log(MAE)', fontsize=16)
    #axs[0,1].set_title(f'Mean Absolute Error: {period}')
    for patch, color in zip(bp_mae['boxes'], np.arange(len(labels))):
        patch.set_facecolor(cmap(color /len(labels)))
    for median in bp_mae['medians']:
        median.set_color(median_color)
    
    bp_r2 = axs[2].boxplot(data_r2, patch_artist=True)
    axs[2].set_xticks(list(np.arange(1,len(labels)+1)))
    axs[2].set_xticklabels(labels, fontsize=16, rotation = 45)
    axs[2].set_ylabel('R2', fontsize=16)
    axs[2].set_ylim((0.93,1.01))
    #axs[1,0].set_title(f'R2 Score: {period}, truncated at 0')
    for patch, color in zip(bp_r2['boxes'], np.arange(len(labels))):
        patch.set_facecolor(cmap(color /len(labels)))
    for median in bp_r2['medians']:
        median.set_color(median_color)
       
    plt.tight_layout()
    plt.savefig(os.path.join(save_to, f'boxplot_leadtime_scores_{data}_{target}.pdf'))
    plt.show()

    
def plot_scores_temporal_targetwise_cumulative(scores_temporal_targetwise_cumulative, score, ax = None, save_to = None, file = ''):
    
    if ax is None:
        # Create a new figure and axes with cartopy projection if not provided
        fig, ax = plt.subplots(figsize=(7, 7))
        own_figure = True
    else:
        # Use the provided axes and assume the calling code handles figure creation
        own_figure = False
           
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.plot(scores_temporal_targetwise_cumulative['swvl1'][score], label='swvl1', color = 'lightblue')
    ax.plot(scores_temporal_targetwise_cumulative['swvl2'][score], label='swvl2', color = 'blue')
    ax.plot(scores_temporal_targetwise_cumulative['swvl3'][score], label='swvl3', color = 'darkblue')
    ax.plot(scores_temporal_targetwise_cumulative['stl1'][score], label='st1', color = 'lightgreen')
    ax.plot(scores_temporal_targetwise_cumulative['stl2'][score], label='stl2', color = 'green')
    ax.plot(scores_temporal_targetwise_cumulative['stl3'][score], label='stl3', color = 'darkgreen')
    ax.plot(scores_temporal_targetwise_cumulative['snowc'][score], label='snowc', color = 'goldenrod')
    ax.set_ylabel(score)
    ax.set_xlabel("Lead time")
    #ax.set_ylim((0.8,1.02))
    ax.set_title(file)
    ax.legend()
    
    if save_to is not None:
        plt.savefig(os.path.join(save_to, f'{score}_temporally_integrated_{file}.pdf'))

        
def boxplot_scores_lstmDEV(xgb_scores, mlp_scores, 
                         lstm1_scores, lstm2_scores, lstm3_scores, lstm4_scores, 
                         log = False, save_to = None):
    
    if log:
        data_rmse = [ np.log(xgb_scores['rmse']),  np.log(mlp_scores['rmse']), 
                     np.log(lstm1_scores['rmse']), np.log(lstm2_scores['rmse']), 
                     np.log(lstm3_scores['rmse']), np.log(lstm4_scores['rmse'])]
        data_mae = [ np.log(xgb_scores['mae']),  np.log(mlp_scores['mae']), 
                     np.log(lstm1_scores['mae']), np.log(lstm2_scores['mae']), 
                     np.log(lstm3_scores['mae']), np.log(lstm4_scores['mae'])]
        data_r2 = [ xgb_scores['r2'],  mlp_scores['r2'],
                     lstm1_scores['r2'], lstm2_scores['r2'],
                    lstm3_scores['r2'], lstm4_scores['r2']]
    else:
        data_rmse = [ xgb_scores['rmse'],  mlp_scores['rmse'],
                     lstm1_scores['rmse'], lstm2_scores['rmse'],
                    lstm3_scores['rmse'], lstm4_scores['rmse']]
        data_mae = [ xgb_scores['mae'],  mlp_scores['mae'],
                     lstm1_scores['mae'], lstm2_scores['mae'],
                    lstm3_scores['mae'], lstm4_scores['mae']]
        data_r2 = [ xgb_scores['r2'],  mlp_scores['r2'],
                     lstm1_scores['r2'], lstm2_scores['r2'],
                    lstm3_scores['r2'], lstm4_scores['r2']]
        
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['lightgreen', 'yellow', 'darkblue','blue', 'lightblue', 'aqua']
    median_color = 'black'
    
    bp_rmse = axs[0].boxplot(data_rmse, patch_artist=True)
    axs[0].set_xticks([1, 2, 3, 4, 5, 6])
    axs[0].set_xticklabels(["XGB", "MLP", "LSTM_basic", "LSTM_dz",  "LSTM_emb", "LSTM_mlpEN"], fontsize=14, rotation=45)
    axs[0].set_ylabel('Log(RMSE)', fontsize=16)
    #axs[0,0].set_title(f'Root Mean Squared Error: {period}')
    for patch, color in zip(bp_rmse['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp_rmse['medians']:
        median.set_color(median_color)
    
    bp_mae = axs[1].boxplot(data_mae, patch_artist=True)
    axs[1].set_xticks([1, 2, 3, 4, 5, 6])
    axs[1].set_xticklabels(["XGB", "MLP", "LSTM_basic", "LSTM_dz",  "LSTM_emb", "LSTM_mlpEN"], fontsize=14, rotation=45)
    axs[1].set_ylabel('Log(MAE)', fontsize=16)
    #axs[0,1].set_title(f'Mean Absolute Error: {period}')
    for patch, color in zip(bp_mae['boxes'], colors):
        patch.set_facecolor(color)
    for median in bp_mae['medians']:
        median.set_color(median_color)
    
    #bp_r2 = axs[1,0].boxplot(data_r2, patch_artist=True)
    #axs[1,0].set_xticks([1, 2, 3])
    #axs[1,0].set_xticklabels(["XGB", "MLP", "LSTM"], fontsize=16)
    #axs[1,0].set_ylabel('R2', fontsize=16)
    #axs[1,0].set_ylim((0.98,1.001))
    #axs[1,0].set_title(f'R2 Score: {period}, truncated at 0')
    #for patch, color in zip(bp_r2['boxes'], colors):
    #    patch.set_facecolor(color)
    #for median in bp_r2['medians']:
    #    median.set_color(median_color)

       
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, f'boxplot_scores_lstmDEV.pdf'))
    plt.show()
 

from datetime import datetime

def make_ailand_plot_combined(y_val, preds_xgb, preds_mlp, preds_lstm, target_size, period, save_to=None, filename = 'ailand_plot_combined.pdf'):
    """
    Ref: Ewan Pinnington ()
    """
    
    fig = plt.figure(figsize=(13, 8))
    ax1 = plt.subplot(331)
    ax2 = plt.subplot(332)
    ax3 = plt.subplot(333)
    ax4 = plt.subplot(334)
    ax5 = plt.subplot(335)
    ax6 = plt.subplot(336)
    ax7 = plt.subplot(337)

    def ailand_plot(idx, ax, ylabel, ax_title, val_date = "2020-09-05", test_date="2022-01-01"):
        """Plotting function for the ec-land database and ai-land model output

        :param var_name: parameter variable name
        :param ax: the axes to plot on
        :param ylabel: y-label for plot
        :param ax_title: title for plot
        :param test_date: date to plot vertical line (train/test split), defaults to "2022-01-01"
        :return: plot axes
        """
        t_eval = preds_mlp.shape[0]-1
        
        # ax.plot(y_preds[:, idx], label="rf-land")
        
        ax.plot(period[:t_eval], y_val[:t_eval, :, -target_size + idx], label="ECland", color="darkblue", linewidth=1.5)
        ax.plot(period[:t_eval], preds_xgb[:t_eval, :, -target_size + idx], label="XGB", color="green", alpha=0.6, linewidth=1.5)
        ax.plot(period[:t_eval], preds_mlp[:t_eval, :, -target_size + idx], label="MLP", color="goldenrod", alpha=0.6, linewidth=1.5)
        ax.plot(period[:t_eval], preds_lstm[:t_eval, :, -target_size + idx], label="LSTM", color="blue", alpha=0.6, linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(ax_title, fontsize=16)
        #ax.axvline(datetime.strptime(val_date, "%Y-%m-%d"), color="k", linestyle="--")
        #ax.axvline(datetime.strptime(test_date, "%Y-%m-%d"), color="k", linestyle="-")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # This removes duplicates.
        ax.legend(by_label.values(), by_label.keys())
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        return ax

    ailand_plot(0, ax1, "Soil Moisture (m3 m-3)", "Soil Moisture Layer 1")
    ailand_plot(1, ax2, "Soil Moisture (m3 m-3)", "Soil Moisture Layer 2")
    ailand_plot(2, ax3, "Soil Moisture (m3 m-3)", "Soil Moisture Layer 3")
    ailand_plot(3, ax4, "Soil Temperature (K)", "Soil Temperature Layer 1")
    ailand_plot(4, ax5, "Soil Temperature (K)", "Soil Temperature Layer 2")
    ailand_plot(5, ax6, "Soil Temperature (K)", "Soil Temperature Layer 3")
    ailand_plot(6, ax7, "Snow Cover Fraction (-)", "Snow Cover Fraction")

    fig.tight_layout()
    if save_to is not None:
        plt.savefig(os.path.join(save_to, filename))
