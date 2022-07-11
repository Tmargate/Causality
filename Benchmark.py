import seaborn as sns

import pandas as pd
from utils import *
import numpy as np
from scipy.linalg.special_matrices import toeplitz
from numpy.random import binomial, multivariate_normal
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import Tensor
import torchtuples as tt


from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
#import lightgbm as lgbm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

        


class Benchmark:
    
    
    def __init__(self, simu):
        
        self.simu = simu
        self.n_features = simu.n_features
        self.n_classes = simu.n_classes
        self.n_samples = simu.n_samples
        self.wd_para = simu.wd_para
        self.beta = simu.beta
        self.coef_tt = simu.coef_tt
        self.X = simu.X_shift
        self.tt = simu.tt
        self.id = simu.id
        self.data_sim = simu.data_sim
        self.function_type = simu.function_type
        
        
        
    def prep_bench(self, test_size = 0.25):
        
        features = self.X
        index = np.arange(0,self.n_samples,1)
        treatment = self.tt.reshape(self.n_samples,1)
        
        self.treatment = treatment
        
        test_size = int(self.n_samples*test_size)
        index_test = index[0:test_size]
        index_train = index[test_size:]
        index_treated = index[(treatment[index] == 1).squeeze()]
        index_untreated = index[(treatment[index] == 0).squeeze()]
        index_treated_train = np.intersect1d(index_treated, index_train)
        index_untreated_train = np.intersect1d(index_untreated, index_train)
        
        self.index_test = index_test
        self.index_train = index_train
        self.index_treated = index_treated
        self.index_untreated = index_untreated
        self.index_treated_train = index_treated_train
        self.index_untreated_train = index_untreated_train
        
        outcome_f = np.asarray(self.data_sim.loc[:,"Y_f"]).reshape(self.n_samples,1)       
        
        self.outcome_f = outcome_f
        
        modele_prscore = linear_model.LogisticRegression(penalty='none',solver='newton-cg')  #regression logisitque pour estimer les propensity_score
        modele_prscore.fit(features,treatment)   #on entraine le modèle
        propensity_logit = features.dot(modele_prscore.coef_.T) + modele_prscore.intercept_
        propensity_logit = propensity_logit.reshape(self.n_samples,)
        propensity_score = sigmoid(propensity_logit).reshape(self.n_samples,1)
        weight = 1/(treatment * propensity_score + (1-treatment) * (1 - propensity_score))
        weight = weight.squeeze()
        
        self.weight = weight
        
        
        
    def predict_one_model(self, model, type_model, weight=True, fit_param_classcaus = None):
        
        index_test = self.index_test
        index_train = self.index_train
        
        features = self.X
        treatment = self.treatment
        treatment_0_test = np.zeros(len(index_test)).reshape(len(index_test),1) 
        treatment_1_test = np.ones(len(index_test)).reshape(len(index_test),1)
        
        features_treatment = np.concatenate((features, treatment), axis =1)
        
        features_train = features_treatment[index_train,:]
        
        features_test = features[index_test,:]
        features_test_0 = np.concatenate((features_test, treatment_0_test), axis = 1)
        features_test_1 = np.concatenate((features_test, treatment_1_test), axis = 1)
        
        outcome_f_train = self.outcome_f[index_train]
        
        weight_train = self.weight[self.index_train]
        
        model_ = model
        
        if type_model =="classcaus":
            
            model_.fit(input=features_train, target=outcome_f_train, **fit_param_classcaus)
            
        else :
            if weight==True:
                model_.fit(X = features_train, y = outcome_f_train, sample_weight = weight_train)
            elif weight==False:
                model_.fit(X = features_train, y = outcome_f_train)
        
        if type_model == "regressor":
            predict_proba_p0 = model_.predict(features_test_0)
            predict_class_p0 = predict_proba_p0 > 0.5
            predict_proba_p1 = model_.predict(features_test_1)
            predict_class_p1 = predict_proba_p1 > 0.5
            
        elif type_model == "classifier":
            predict_proba_p0 = model_.predict_proba(features_test_0)[:,1]
            predict_class_p0 = predict_proba_p0 > 0.5
            predict_proba_p1 = model_.predict_proba(features_test_1)[:,1]
            predict_class_p1 = predict_proba_p1 > 0.5  
            
        elif type_model == "classcaus":
            predict_proba_p0 = model_.predict_proba(features_test_0)
            predict_class_p0 = predict_proba_p0 > 0.5
            predict_proba_p1 = model_.predict_proba(features_test_1)
            predict_class_p1 = predict_proba_p1 > 0.5  
        
        self.predict_proba_p1 = predict_proba_p1
        self.predict_class_p1 = predict_class_p1
        self.predict_proba_p0 = predict_proba_p0
        self.predict_class_p0 = predict_class_p0
        
        
    def predict_two_models(self, model0, model1, type_model, weight=True):
            
            
        index_test = self.index_test
        index_untreated_train = self.index_untreated_train
        index_treated_train = self.index_treated_train
        
        features = self.X
        
        features_untreated_train = features[index_untreated_train,:]
        features_treated_train = features[index_treated_train,:]

        features_test = features[index_test,:]
        
        outcome_f_untreated_train  = self.outcome_f[index_untreated_train]
        outcome_f_treated_train  = self.outcome_f[index_treated_train]
        
        weight_untreated_train = self.weight[index_untreated_train]
        weight_treated_train = self.weight[index_treated_train]
        
        model_0 = model0
        model_1 = model1
        
        if weight==True:
            model_0.fit(X= features_untreated_train, y= outcome_f_untreated_train, sample_weight=weight_untreated_train)   
            model_1.fit(X= features_treated_train, y= outcome_f_treated_train, sample_weight=weight_treated_train)
            
        elif weight==False:
            model_0.fit(X= features_untreated_train, y= outcome_f_untreated_train)   
            model_1.fit(X= features_treated_train, y= outcome_f_treated_train)
        
        if type_model == "regressor":
            predict_proba_p0 = model_0.predict(features_test)
            predict_class_p0 = predict_proba_p0 > 0.5
            predict_proba_p1 = model_1.predict(features_test)
            predict_class_p1 = predict_proba_p1 > 0.5
            
        elif type_model =="classifier":
            predict_proba_p0 = model_0.predict_proba(features_test)[:,1]
            predict_class_p0 = predict_proba_p0 > 0.5
            predict_proba_p1 = model_1.predict_proba(features_test)[:,1]
            predict_class_p1 = predict_proba_p1 > 0.5  
        
        
        
        self.predict_proba_p1 = predict_proba_p1
        self.predict_class_p1 = predict_class_p1
        self.predict_proba_p0 = predict_proba_p0
        self.predict_class_p0 = predict_class_p0
        
        
    def benchmark_one_model(self, model, type_model, name_model="model", weight=True,  fit_param_classcaus = None):
        
        index_test = self.index_test
        
        outcome_0_test = np.asarray(self.data_sim.loc[index_test,"Y_0"])
        outcome_1_test = np.asarray(self.data_sim.loc[index_test,"Y_1"])
        
        self.predict_one_model(model = model, type_model = type_model, weight = weight,
                               fit_param_classcaus = fit_param_classcaus)
        
        predict_proba_p0 = self.predict_proba_p0
        predict_proba_p1 = self.predict_proba_p1
        simu_proba_p0 = np.asarray(self.data_sim.loc[index_test,"pi_0"])
        simu_proba_p1 = np.asarray(self.data_sim.loc[index_test,"pi_1"])
        
        predict_class_p0 = self.predict_class_p0
        predict_class_p1 = self.predict_class_p1
        
        accuracy_pred_0 = accuracy_score(outcome_0_test, predict_class_p0)
        #auc_pred_0 = roc_auc_score(outcome_0_test, predict_proba_p0)
        MSE_proba_0 = mean_squared_error(simu_proba_p0, predict_proba_p0)
        
        accuracy_pred_1 = accuracy_score(outcome_1_test, predict_class_p1)
        #auc_pred_1 = roc_auc_score(outcome_1_test, predict_proba_p1)
        MSE_proba_1 = mean_squared_error(simu_proba_p1, predict_proba_p1)
        
        cate_predict = predict_proba_p1 - predict_proba_p0
        cate_simu  = simu_proba_p1 - simu_proba_p0
        PEHE = np.sqrt(mean_squared_error(cate_simu, cate_predict))
        
        sign_cate_predict = (cate_predict >= 0)
        sign_cate_simu = (cate_simu >= 0)
        accu_sign_cate = np.mean(sign_cate_predict == sign_cate_simu)
        
        self.cate_predict = cate_predict
        self.cate_simu = cate_simu
        
        score_list = np.ones(6)
        score_list[0] = accuracy_pred_0
        score_list[1] = accuracy_pred_1
        #score_list[2] = auc_pred_0
        #score_list[3] = auc_pred_1
        score_list[2] = MSE_proba_0
        score_list[3] = MSE_proba_1
        score_list[4] = PEHE
        score_list[5] = accu_sign_cate
        score_list = score_list.reshape(1,6)
        
        name_columns = ["accu_0","accu_1","MSE_p0", "MSE_p1","PEHE","accu_sign_cate"]
        pd_benchmark = pd.DataFrame(data=score_list, columns=name_columns, index=[name_model])
        
        self.accuracy_pred_0 = accuracy_pred_0
        self.accuracy_pred_1 = accuracy_pred_1
        #self.auc_pred_0 = auc_pred_0
        #self.auc_pred_1 = auc_pred_1
        self.MSE_proba_0 = MSE_proba_0
        self.MSE_proba_1 = MSE_proba_1
        self.PEHE = PEHE
        self.accu_sign_cate = accu_sign_cate
    
    
        return (pd_benchmark)
    
    
    def benchmark_two_models(self, model0, model1, type_model, name_models = "model0 and model1", weight=True):
        
            
        index_test = self.index_test
            
        outcome_0_test = np.asarray(self.data_sim.loc[index_test,"Y_0"])
        outcome_1_test = np.asarray(self.data_sim.loc[index_test,"Y_1"])
        
        self.predict_two_models(model0 = model0, model1 = model1, type_model = type_model, weight=weight)
        
        predict_proba_p0 = self.predict_proba_p0
        predict_proba_p1 = self.predict_proba_p1
        simu_proba_p0 = np.asarray(self.data_sim.loc[index_test,"pi_0"])
        simu_proba_p1 = np.asarray(self.data_sim.loc[index_test,"pi_1"])
        
        predict_class_p0 = self.predict_class_p0
        predict_class_p1 = self.predict_class_p1
        
        accuracy_pred_0 = accuracy_score(outcome_0_test, predict_class_p0)
        #auc_pred_0 = roc_auc_score(outcome_0_test, predict_proba_p0)
        MSE_proba_0 = mean_squared_error(simu_proba_p0, predict_proba_p0)
        
        accuracy_pred_1 = accuracy_score(outcome_1_test, predict_class_p1)
        #auc_pred_1 = roc_auc_score(outcome_1_test, predict_proba_p1)
        MSE_proba_1 = mean_squared_error(simu_proba_p1, predict_proba_p1)
        
        cate_predict = predict_proba_p1 - predict_proba_p0
        cate_simu  = simu_proba_p1 - simu_proba_p0
        PEHE = np.sqrt(mean_squared_error(cate_simu, cate_predict))
        
        sign_cate_predict = (cate_predict >= 0)
        sign_cate_simu = (cate_simu >= 0)
        accu_sign_cate = np.mean(sign_cate_predict == sign_cate_simu)
        
        self.cate_predict = cate_predict
        self.cate_simu = cate_simu
        
        score_list = np.ones(6)
        score_list[0] = accuracy_pred_0
        score_list[1] = accuracy_pred_1
        #score_list[2] = auc_pred_0
        #score_list[3] = auc_pred_1
        score_list[2] = MSE_proba_0
        score_list[3] = MSE_proba_1
        score_list[4] = PEHE
        score_list[5] = accu_sign_cate
        score_list = score_list.reshape(1,6)
        
        name_columns = ["accu_0","accu_1","MSE_p0", "MSE_p1","PEHE","accu_sign_cate"]
        pd_benchmark = pd.DataFrame(data=score_list, columns=name_columns, index=[name_models])
        
        self.accuracy_pred_0 = accuracy_pred_0
        self.accuracy_pred_1 = accuracy_pred_1
        #self.auc_pred_0 = auc_pred_0
        #self.auc_pred_1 = auc_pred_1
        self.MSE_proba_0 = MSE_proba_0
        self.MSE_proba_1 = MSE_proba_1
        self.PEHE = PEHE
        self.accu_sign_cate = accu_sign_cate
    
    
        return (pd_benchmark)