import numpy as np
import scipy.special as special
from scipy.stats import pearsonr, norm, entropy, ttest_ind, f, bartlett
import sklearn.metrics
#import CRPS.CRPS as pscore


def anomaly(y_hat, climatology):
    
    return y_hat - climatology
    
def standardized_anomaly(y_hat, climatology, climatology_std):
    
    return anomaly(forecast, climatology)/climatology_std

def anomaly_correlation(forecast, reference, climatology):
    
    anomaly_f = anomaly(forecast, climatology)
    anomaly_r = anomaly(reference, climatology)
    
    msse = np.mean(anomaly_f * anomaly_r)
    act = np.sqrt(np.mean(anomaly_f**2) * np.mean(anomaly_r**2))
    
    return msse/act


def rmse_saturation(reference, climatology):
    
    anomaly_r = anomaly(reference, climatology)
    A_r_squared =  np.mean(anomaly_r**2)
    A_r = np.sqrt(A_r_squared)
    saturation_level = A_r*np.sqrt(2)
    
    return saturation_level
    
def absolute_differences(reference, ensemble, mean = False):
    absolute_differences = abs(np.subtract(reference, ensemble))
    return absolute_differences

def rolling_bias(reference, ensemble):

    b = []
    for i in range(reference.shape[1]):
        b.append(np.mean(np.subtract(reference[0,:i], ensemble[:,:i]), axis=1))
    b = np.transpose(np.array(b))

    return b

def rolling_mse(reference, ensemble):

    """
    Calculates forecast rmse for a time series of predictions by stepwise adding the next time step.
    Change this to a moving window? Or pointwise?
    :param preds: predicted time series
    :return: time series of rmse
    """
    mses = []
    for i in range(reference.shape[1]):
        mse = np.mean(np.subtract(reference[0,:i], ensemble[:, :i])**2, axis=1)
        mses.append(mse)
    return np.transpose(np.array(mses))

def rolling_rmse(reference, ensemble, standardized = False):

    """
    Calculates forecast rmse for a time series of predictions by stepwise adding the next time step.
    Change this to a moving window? Or pointwise?
    :param preds: predicted time series
    :return: time series of rmse
    """
    rmses = []
    for i in range(reference.shape[1]):
        rmse = np.sqrt(np.mean(np.subtract(reference[0,:i], ensemble[:, :i])**2, axis=1))
        rmses.append(rmse)
    return np.transpose(np.array(rmses))

def rolling_corrs(reference, ensemble, window = 3):
    """
    Rolling correlations between true and predicted dynamics in a moving window.
    Change to cross-correlation?
    :param obs: np.vector. true dynamics
    :param preds: np.array. ensemble predictions.
    :param test_index: int. Where to start calculation
    :param window: int. Size of moving window.
    :return: array with correlations. shape:
    """
    corrs = [[pearsonr(reference[0,j:j+window], ensemble[i,j:j+window])[0] for i in range(ensemble.shape[0])] for j in range(reference.shape[1]-window)]
    corrs = np.transpose(np.array(corrs))

    return corrs


def rolling_crps(reference, ensemble):
    """
    """
    if len(ensemble.shape) == 1:
        ensemble = np.expand_dims(ensemble, 0)

    crps = np.array([pscore(ensemble[:,i], reference[:,i]).compute() for i in range(reference.shape[1])]).squeeze()
    crps = crps[:,0]
    if crps.dtype != 'float32':
        crps = crps.astype('float32')

    return crps  # return only the basic crps (p-score returns three crps types)

def rolling_rsquared(reference, ensemble):

    r_sq = []
    for j in range(1, ensemble.shape[1]):
        r_sq.append([sklearn.metrics.r2_score(reference[:, :j].transpose(), ensemble[i, :j]) for i in range(ensemble.shape[0])])
    r_sq = np.array(r_sq).transpose()

    return r_sq

def rmse(reference, ensemble, standardized = False):
    if standardized:
        return np.sqrt(np.mean(np.subtract(reference,ensemble)**2, axis=0))/ (np.max(reference) - np.min(reference))
    else:
        return np.sqrt(np.mean(np.subtract(reference,ensemble)**2, axis=0))

def mse(reference, ensemble, ensemble_mean=True):
    if ensemble_mean:
        mse = np.mean(np.subtract(reference, ensemble)**2, axis=0)

    return mse

def column_wise_rmse(yhat, yosm):
    if yhat.shape[1] != yosm.shape[1]:
        raise ValueError("The matrices must have the same number of columns.")

    squared_diff = np.square(yosm - yhat)
    rmse = np.sqrt(np.mean(squared_diff, axis=0))
    return rmse

def column_wise_mae(yhat, yosm):
    if yhat.shape[1] != yosm.shape[1]:
        raise ValueError("The matrices must have the same number of columns.")

    mae = np.mean(np.abs(np.subtract(yhat, yosm)), axis=0)
    return mae

def squared_error_SNR(obs, pred):
    """
    The squared error based SNR, an estimator of the true expected SNR. (Czanner et al. 2015, PNAS)
    The signal is the reduction in expected prediction error by using the model that generated pred.
    Independent of sample size!
    """

    EPE_mean = np.dot(np.transpose(obs - np.mean(obs)), (obs - np.mean(obs))) # returns a scalar
    EPE_model = np.dot(np.transpose(obs - pred), (obs - pred)) # also returns a scalar
    signal = (EPE_mean - EPE_model)
    noise = EPE_model

    return signal/noise


def var_based_SNR(obs, pred, inital_uncertainty):
    """
    The squared error based SNR, an estimator of the true expected SNR at perfect knowledge of parameters.
    The signal is the reduction in expected prediction error by using the model that generated pred.
    Dependent on sample size (decreases with sample size)

    # This may be more suited for the perfect model scenario
    # but I am not going to use it until I am sure of what the individual parts are
    """
    signal = np.dot(np.transpose(pred - np.mean(obs)), (pred - np.mean(obs)))
    noise = len(obs) * inital_uncertainty ** 2

    return signal / noise

def raw_SNR(pred, var = False):
    # tSNR raw SNR or timeseries SNR: mean(timeseries) / var(timeseries)
    # tsnr increases with sample size (see sd).

    signal = np.mean(pred, axis=1)
    if var:
        noise = np.mean(pred**2, axis=1) - np.mean(pred, axis=1)**2 # np.std(pred, axis=1)#1/pred.shape[0]*np.sum(np.subtract(pred, mu)**2, axis=0)
    else:
        noise = np.std(pred,axis=1)

    return signal/noise

def raw_CNR(obs, pred, squared = False):
    """
    CNR - contrast to noise ratio: mean(condition-baseline_1D) / std(baseline_1D)
    This is basically the same as the square-error-based SNR?
    Transfered, we have the model as the baseline_1D and the mean as condition.
    tsnr increases with sample size (see sd).
    """
    signal = np.mean((pred - np.mean(obs))) # returns a scalar
    noise = np.std(obs)
    if squared:
        return signal**2/noise**2, signal**2, noise**2
    else:
        return signal/noise, signal, noise


def rolling_CNR(obs, pred, squared = False):
    """
    CNR - contrast to noise ratio: mean(condition-baseline_1D) / std(baseline_1D)
    This is basically the same as the square-error-based SNR?
    Transfered, we have the model as the baseline_1D and the mean as condition.
    tsnr increases with sample size (see sd).
    """
    cnrs = []
    for i in range(2,pred.shape[1]):
        signal = np.mean(np.subtract(pred[:,:i],np.mean(obs[:,:i])), axis=1) # returns a scalar
        noise = np.std(obs[:,:i])
        if squared:
            cnr = signal**2/noise**2
        else:
            cnr = signal/noise
        cnrs.append(cnr)

    return np.transpose(np.array(cnrs))

def bs_sampling(obs, pred, snr, samples=100):

    its = obs.shape[1]
    arr = np.zeros((its, samples))

    for j in range(samples):

        obs_ind, pred_ind = np.random.randint(obs.shape[0], size=1), np.random.randint(pred.shape[0], size=1)
        x_obs = obs[obs_ind].flatten()
        x_pred = pred[pred_ind].flatten()

        for i in range(its):

            if snr == "cnr":
                arr[i, j] = raw_CNR(x_obs[:i + 2], x_pred[:i + 2])[0]
            elif snr == "ss-snr":
                arr[i, j] = squared_error_SNR(x_obs[:i + 2], x_pred[:i + 2])

    return arr

def tstatistic_manual(x_sample, H0):
    """
    Student's t-test. Two-sided.
    """
    df = x_sample.shape[0]-1
    v = np.var(x_sample, axis=0, ddof=1)
    denom = np.sqrt(v/df)
    t = np.divide((x_sample.mean(axis=0)-H0),denom)
    pval = special.stdtr(df, -np.abs(t))*2
    return t, pval

def tstatistic_scipy(x, y):
    """
    Computes tstatistics between two sets. Returns t-statistics and p-values.
    :param x: np array
    :param y: np array
    :return: tstats, pvalues
    """
    ttest_results = ttest_ind(x.flatten(), y.flatten(), equal_var=False)
    tstats = ttest_results.statistic
    pvalues = ttest_results.pvalue

    return tstats, pvalues

def entropy_manual(p, q, integral = True):
    """

    :param p:
    :param q:
    :param integral:
    :return:
    """
    prob_frac = np.round(p/q, 50)
    prob_frac[prob_frac == 0] = 1e-50
    RE = np.sum(p*np.log(prob_frac)) if integral else p*np.log(prob_frac)
    return RE

def entropy(x, y):

    """
    Computes relative entropy between two sets, x and y.
    X and y are assumed normal, i.e. mean and std are fitted to generate probablity density function.
    :param x: np array
    :param y: np array
    :return: entropy as in scipy.stats.
    """
    mu, std = norm.fit(x.flatten())
    xmin, xmax = x.min(), x.max()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    mu, std = norm.fit(y.flatten())
    ymin, ymax = y.min(), y.max()
    y = np.linspace(ymin, ymax, 100)
    q = norm.pdf(y, mu, std)

    return entropy(p, q)


def nash_sutcliffe(observed, modeled):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    """
    observed = np.array(observed)
    modeled = np.array(modeled)
    mean_observed = np.mean(observed)

    # Calculate sum of squared differences between observed and modeled values
    ss_diff = np.sum((observed - modeled) ** 2)
    # Calculate sum of squared differences between observed and mean of observed values
    ss_total = np.sum((observed - mean_observed) ** 2)

    # Nash-Sutcliffe Efficiency
    nse = 1 - (ss_diff / ss_total)

    return nse

def rolling_nash_sutcliffe(reference, ensemble, window=2):
    nse = [[nash_sutcliffe(reference[j:j + window], ensemble[i, j:j + window]) for i in range(ensemble.shape[0])] for
             j in range(ensemble.shape[1] - window)]
    nse = np.transpose(np.array(nse))
    return nse


def fstat(forecast, observation, bartlett_test = False):

    if bartlett_test:
        bs = np.array([bartlett(forecast[:, i].squeeze(), observation[:,:150].squeeze()) for i in range(forecast.shape[1])])
        stats = bs[:,0]
        pvals = bs[:,1]
    else:
        vars = [np.var(forecast[:, i]) for i in range(forecast.shape[1])]
        var_obs = np.var(observation)
        stats = [(vars[i] / var_obs) for i in range(len(vars))] # ppp = [(1-vars[i]/var_true) for i in range(len(vars))]
        df1, df2 = forecast.shape[0] - 1, forecast.shape[1] - 1
        pvals = [f.cdf(stats[i], df1, df2) for i in range(len(stats))]
    return stats, pvals

def tstat_inverse(t, samples):

    return t/np.sqrt((samples-2+t**2))
