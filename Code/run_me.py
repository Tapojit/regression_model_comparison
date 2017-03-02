# run_me.py module

import kaggle

# Assuming you are running run_me.py from the Submission/Code directory, otherwise the path variable will be different for you
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor

#Air Foil data
path = '../../Data/AirFoil/'



#final pipeline
#path-directory of dataset
#param_grid-dictionary of hyperparameters to be optimized and range of their values
#nfs-n features to select. None if not applicable
#data_set-name of dataset
def pipe_line_final(path, param_grid, nfs, data_set):
    if data_set=='Blog':
        train_m = np.load(path + 'train.npy')
        X_all=train_m[:,0:train_m.shape[1]-1]
        Y_all=train_m[:,-1]
        f=SelectKBest(f_regression)
        X_new=f.fit_transform(X_all, Y_all)
        train_x, train_y=shuffle(X_new, Y_all)
        selected_params=f.get_support(indices=True)
        
       
    
        
        reg=GridSearchCV(GradientBoostingRegressor(), param_grid, scoring='neg_mean_squared_error')
        reg.fit(train_x,train_y)
        
        print "For ", data_set, "- best params: ", reg.best_params_, "Feature_subset: ", f.get_support(indices=True)
        
        test=np.load(path + 'test_distribute.npy')
        test_x=test[:, 0:test.shape[1]-1]
        tr=[]
        for i in range(0, 280):
            if i in selected_params:
                tr.append(True)
            else: tr.append(False) 
        tr=np.array(tr)
        X_masked=test_x[:,tr]
        predictions=reg.predict(X_masked)
        kaggle.kaggleize(predictions, "../Predictions/BlogFeedback/test.csv")
    
    else:
        train = np.load(path + 'train.npy')
        test = np.load(path + 'test_private.npy')
        train_x, train_y=shuffle(train[:, 0:train.shape[1]-1], train[:, -1])
        test_x = test[:, 0:test.shape[1]-1]
        test_y = test[:, -1]
        rfe=RFE(DecisionTreeRegressor(), n_features_to_select=nfs, step=2)
        reg=GridSearchCV(rfe, param_grid)
        reg.fit(train_x,train_y)
        print "For ", data_set, "- best params: ", reg.best_params_, "Feature_subset: ", reg.best_estimator_.ranking_, "RMSE: ", np.sqrt(mean_squared_error(test_y,reg.predict(test_x)))









#initial pipeline
#path-directory of dataset
#test_y_p-boolean;True if labels are present in test file
#param_grid-dictionary of hyperparameters to be optimized and range of their values
#nfs-n features to select. None if not applicable
#data_set-name of dataset
def pipe_line(path, test_y_p, param_grid, nfs, data_set):
    if test_y_p:
        train = np.load(path + 'train.npy')
        test = np.load(path + 'test_private.npy')
    else:
        train_m = np.load(path + 'train.npy')
        train=train_m[:(len(train_m)/2)]
        test = train_m[(len(train_m)/2):]
        
        
        
        
    train_x, train_y=shuffle(train[:, 0:train.shape[1]-1], train[:, -1])
    test_x = test[:, 0:test.shape[1]-1]
    test_y = test[:, -1]
    rfe=RFE(DecisionTreeRegressor(), n_features_to_select=nfs, step=2)
    reg=GridSearchCV(rfe, param_grid)
    reg.fit(train_x,train_y)
    print "For ", data_set, "- best params: ", reg.best_params_, "Feature_subset: ", reg.best_estimator_.ranking_, "RMSE: ", np.sqrt(mean_squared_error(test_y,reg.predict(test_x)))
    if not test_y_p:
        test_f=np.load(path + 'test_distribute.npy')
        predictions=reg.predict(test_f[:, 0:test.shape[1]-1])
        kaggle.kaggleize(predictions, "../Predictions/BlogFeedback/test.csv")
        
        
#Array of range of min_impurity_split       
m_i_s=[1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
#n features to select
nfs=4
#Array of range of max_features
m_f=list(range(1,nfs+1))
#feature grid
diction={'estimator__max_features': m_f,'estimator__min_impurity_split': m_i_s}
pipe_line_final(path, diction, nfs, 'AirFoil')
   

#1.6) Default hyperparameters used (Note- only for DecisionTreeRegressor)
#path-directory of dataset
#data_set=name of dataset
def default_hpm(path, data_set):
    train = np.load(path + 'train.npy')
    test = np.load(path + 'test_private.npy')
    train_x, train_y=shuffle(train[:, 0:train.shape[1]-1], train[:, -1])
    test_x = test[:, 0:test.shape[1]-1]
    test_y = test[:, -1]
    reg=DecisionTreeRegressor()
    reg.fit(train_x,train_y)
    print "For ", data_set, "-", "RMSE: ", np.sqrt(mean_squared_error(test_y,reg.predict(test_x)))
default_hpm(path, 'AirFoil')

#1.7) Plots 2 line graphs in 1 set of axes.
#hp=array of hyperparameter values
#y1=array of left y-axis values
#y2=array of right y-axis values
#x_lab=x-axis label
#y1_lab=left y-axis label
#y2_lab=right y-axis label
def line_plot(hp, y1, y2, x_lab, y1_lab, y2_lab):
    inds=hp
    values1=y1
    values2=y2
    title='RMSE vs. '+x_lab
    #Plot a line graph
    fig=plt.figure(2, figsize=(10,4))
    ax1=fig.add_subplot(111)
    ax1.plot(inds, values1, 'or-', linewidth=3)
    ax1.set_ylabel(y1_lab, color='r')
    ax1.set_xlabel(x_lab)
    ax1.set_xlim(min(hp),max(hp))
    for tl in ax1.get_yticklabels():
        tl.set_color('r')
    ax1.set_title(title)
    
    
    ax2=ax1.twinx()
    ax2.plot(inds, values2, 'sb-', linewidth=3)
    ax2.set_ylabel(y2_lab, color='b')
    ax2.set_xlim(min(hp),max(hp))
    for tl in ax2.get_yticklabels():
        tl.set_color('b')
    plt.savefig("../Figures/"+title+".png")
    
    print "Line graph image generated"


#1.7) Crossvalidation implementation method (Note-line-line_plot is called here to create line-graphs.)
#path-directory to data set.
#data_set-name of data set
#hp_choice-hyperparameter whose range of values are cross-validated for optimization. Only for 'min_impurity_split' and 'max_features'
def cv_imp(path, data_set, hp_choice):
    train = np.load(path + 'train.npy')
    test = np.load(path + 'test_private.npy')
    train_x, train_y=shuffle(train[:, 0:train.shape[1]-1], train[:, -1])
    test_x = test[:, 0:test.shape[1]-1]
    test_y = test[:, -1]
    if hp_choice=='min_impurity_split':
        m_i_s=[1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
        diction={'min_impurity_split': m_i_s}
        reg=GridSearchCV(DecisionTreeRegressor(max_features=4), diction, scoring='neg_mean_squared_error')

    else:    
        m_f=list(range(1,6))
        diction={'max_features': m_f}
        reg=GridSearchCV(DecisionTreeRegressor(min_impurity_split=1e-11), diction, scoring='neg_mean_squared_error')
    reg.fit(train_x,train_y)
    train_scores, valid_scores=reg.cv_results_['mean_train_score'], reg.cv_results_['mean_test_score']
    for i in range(len(valid_scores)):
        valid_scores[i]=np.sqrt(np.abs(valid_scores[i]))
    for i in range(len(train_scores)):
        train_scores[i]=np.sqrt(np.abs(train_scores[i]))
    print "For ", data_set, "- best params: ", reg.best_params_, "RMSE: ", np.sqrt(mean_squared_error(test_y,reg.predict(test_x)))
    if hp_choice=='min_impurity_split':
        line_plot(m_i_s, valid_scores, train_scores, hp_choice, 'validation_RMSE', 'training_RMSE')
    else: line_plot(m_f, valid_scores, train_scores, hp_choice, 'validation_RMSE', 'training_RMSE')


cv_imp(path, 'AirFoil', 'min_impurity_split')
cv_imp(path, 'AirFoil', 'max_features')


def def_feature_select(path, data_set, nfs):
    train = np.load(path + 'train.npy')
    test = np.load(path + 'test_private.npy')
    train_x, train_y=shuffle(train[:, 0:train.shape[1]-1], train[:, -1])
    test_x = test[:, 0:test.shape[1]-1]
    test_y = test[:, -1]
    
    rfe=RFE(DecisionTreeRegressor(),n_features_to_select=nfs, step=2)
    rfe.fit(train_x,train_y)
    print "For ", data_set, "- feature_subset: ", rfe.ranking_, "RMSE: ", np.sqrt(mean_squared_error(test_y,rfe.predict(test_x)))
def_feature_select(path, 'AirFoil', 4)

#2.1) Dangerous: Takes very long time to run commented out code!!


# m_i_s=[1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
# nfs=4
# m_f=list(range(1,nfs+1))
# diction={'estimator__max_features': m_f,'estimator__min_impurity_split': m_i_s}
# pipe_line('../../Data/BlogFeedback/', False, diction, nfs, 'Blog')


#Blog FeedBack
path2='../../Data/BlogFeedback/'
#Array of range of min_impurity_split 
m_i_s=[1e-09, 1e-10,1e-11,1e-12, 1e-13]
#Array of range of max_features
m_f=list(range(3,8))
#feature grid
diction_2={'max_features': m_f, 'min_impurity_split': m_i_s}

pipe_line_final(path2, diction_2, None, 'Blog')







