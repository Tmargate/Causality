import numpy as np
from sklearn.preprocessing import label_binarize
# from scipy import stats
import torch  # For building the networks
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
def sigmoid(x):
    idx = x > 0
    out = np.empty(x.size)
    out[idx] = 1 / (1. + np.exp(-x[idx]))
    exp_x = np.exp(x[~idx])
    out[~idx] = exp_x / (1. + exp_x)
    return out


def get_data(input):
    x = input[:, :-1]
    t = input[:, -1]
    x = x.clone().detach().float()  # torch.tensor(x).float()
    t = t.clone().detach().float()  # torch.tensor(t).float()
    t = t.view(-1, 1)
    return x, t


def sepr_repr(x):
    mask_1, mask_0 = (x[:, -1] == 1), (x[:, -1] == 0)
    x_1 = x[mask_1]
    x_1 = x_1[:, :-1]
    x_0 = x[mask_0]
    x_0 = x_0[:, :-1]
    m = max(x_0.shape, x_1.shape)
    z0 = torch.zeros(m)
    m0 = x_0.shape[0]
    z0[:m0, ] = x_0
    z1 = torch.zeros(m)
    m1 = x_1.shape[0]
    z1[:m1, ] = x_1
    return z0, z1


class SinkhornDistance(nn.Module):
    """[summary]

    """

    def __init__(self, eps, max_iter, reduction="none"):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = (torch.empty(batch_size,
                          x_points,
                          dtype=torch.float,
                          requires_grad=False).fill_(1.0 / x_points).squeeze())
        nu = (torch.empty(batch_size,
                          y_points,
                          dtype=torch.float,
                          requires_grad=False).fill_(1.0 / y_points).squeeze())

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (self.eps * (torch.log(mu + 1e-8) -
                             torch.logsumexp(self.M(C, u, v), dim=-1)) + u)
            v = (self.eps *
                 (torch.log(nu + 1e-8) -
                  torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) +
                 v)
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost  # , pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin))**p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


"""def KL(a, b):

    a = np.asarray(a, dtype=np.float)
    a = a / np.mean(a)
    b = np.asarray(b, dtype=np.float)
    b = b / np.mean(b)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
"""

def get_x_tt(x_test, t):
            tt = torch.ones(x_test.shape[0], 1) * t
            x_test_t = np.concatenate(
                (x_test[:, :-1], tt.reshape(-1, 1)), axis=1)
            x_test_t = torch.from_numpy(x_test_t).float()
            return x_test_t
        
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    a = a / np.mean(a)
    b = np.asarray(b, dtype=np.float)
    b = b / np.mean(b)
    
    return np.mean(np.abs(a - b))/2.



def std_diff_metric(x, y):
    x = np.asarray(x, dtype=np.float)
    y = np.asarray(y, dtype=np.float)
    return np.std(x - y)
def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def boxplot_F(models_list,d_exp,xlabel='Models',ylabel='',option=0):
    n = len(models_list)
    def f(l):
        l= np.asarray(l)
        return np.abs(l)
    if option == 0:
        input = [f(d_exp[f'{model_name}']['p_test_0']-d_exp[f'{model_name}']['p_pred_0']) for model_name in models_list]
    if option == 1:
        input = [f(d_exp[f'{model_name}']['p_test_1']-d_exp[f'{model_name}']['p_pred_1']) for model_name in models_list]
    """if option == 2:
        input = [d_exp[f'CATE_{model_name}'] for model_name in models_list]"""

    fig = plt.figure(figsize=(18, 10), dpi=100)
    ax = fig.add_subplot(111)
    bp = ax.boxplot(input, widths=0.2, sym='', patch_artist=True)
    plt.setp(bp['caps'], color='blue', alpha=1)
    plt.setp(bp['whiskers'], color='blue', alpha=1)
    plt.setp(bp['means'], color='blue', alpha=1)
    plt.setp(bp['boxes'],
             facecolor= 'blue', alpha=1)
    plt.setp(bp['fliers'], color='blue', alpha=1)
     
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.xticks(np.arange(n) + 1, models_list, rotation=60)
    plt.close()
    return fig

def get_dtv(models_list,d_all):
    dtv = []
    pehe_l = []
    for model_name in models_list:
        dtv_0 = d_all[f'{model_name}']['d_0']['kl']
        dtv_1 = d_all[f'{model_name}']['d_1']['kl']
        pehe = d_all[f'{model_name}']['d_0']['pehe']
        dtv.append(dtv_0+dtv_1)
        pehe_l.append(pehe)
    d= {'Model': models_list,
           'DTV': dtv,
           'PEHE': pehe_l}
    df = pd.DataFrame(data=d)
    return df

    



def dist_F(models_list,d_exp,xlabel='Models',ylabel='',option=0):
    n = len(models_list)
    if option == 0:
        p_test_models = [d_exp[f'{model_name}']['p_test_0'] for model_name in models_list]
        p_pred_models = [d_exp[f'{model_name}']['p_pred_0'] for model_name in models_list]
    if option == 1:
        p_test_models = [d_exp[f'{model_name}']['p_test_1'] for model_name in models_list]
        p_pred_models = [d_exp[f'{model_name}']['p_pred_1'] for model_name in models_list]
    
    list_fig = []
    for model in models_list:
        fig = plt.figure(figsize=(18, 10), dpi=300)
        ax = fig.add_subplot(111)
        sns.distplot(p_test_models[models_list.index(model)], hist=False, kde_kws={'shade': True, 'linewidth': 3}, ax=ax,label= f'Distribution of test p_{option} for model {model}')
        sns.distplot(p_pred_models[models_list.index(model)], hist=False, kde_kws={'shade': True, 'linewidth': 3}, ax=ax,label= f'Distribution of pred p_{option} for model {model}')
        #plt.xlabel(str(xlabel)) 
        plt.ylabel(str(ylabel))
        plt.legend()
        plt.close()
        list_fig.append(fig)
    return list_fig

