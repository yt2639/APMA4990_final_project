
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import metrics
import pickle

def XGBoost(X, y, n_splits=5, random_state=666,
            n_estimators = 350,
            learning_rate = 0.08,
            gamma = 0,
            subsample = 0.75,
            colsample_bytree = 1,
            max_depth = 2,
            min_child_weight = 4,
            silent = 1,
            n_jobs = -1):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prediction_part = []
    index = []
    RMSE = []
    RMSLE = []
    R2 = []
    # threshold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,], y.iloc[test_index,]
        #########################
        # fit your model here
        xgb_raw = XGBRegressor(n_estimators = n_estimators,
                               learning_rate = learning_rate,
                               gamma = gamma,
                               subsample = subsample,
                               colsample_bytree = colsample_bytree,
                               max_depth = max_depth,
                               min_child_weight = min_child_weight,
                               silent = silent,
                               n_jobs = n_jobs)
        xgb_raw.fit(X_train, y_train)
        y_pred_xgb = xgb_raw.predict(X_test)
        # get metrics here
        rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred_xgb))
        rmsle = np.sqrt(np.mean(np.square(abs(np.log1p(y_test) - np.log1p(y_pred_xgb)))))
        r2 = metrics.r2_score(y_test, y_pred_xgb)
        #########################
        # # save model here
        # if rmse>threshold:
        #     with open('saved/xgb.pickle', 'wb') as f:
        #         pickle.dump(xgb_raw, f)
        #     threshold=rmse
        #########################

        #########################
        prediction_part.extend(y_pred_xgb)
        index.extend(test_index)
        RMSE.append(rmse)
        RMSLE.append(rmsle)
        R2.append(r2)

    prediction_all = np.column_stack((index,prediction_part))

    return prediction_all, np.mean(RMSE), np.mean(RMSLE), np.mean(R2)


def KNN(X, y, n_splits=5, random_state=666,n_neighbors=15):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prediction_part = []
    index = []
    RMSE = []
    RMSLE = []
    R2 = []
    # threshold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,], y.iloc[test_index,]
        #########################
        # fit your model here
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        neigh.fit(X_train,y_train)
        Y_pred = neigh.predict(X_test)
        # get metrics here
        rmse = np.sqrt(metrics.mean_squared_error(y_test,Y_pred))
        rmsle = np.sqrt(np.mean(np.square(abs(np.log1p(y_test) - np.log1p(Y_pred)))))
        r2 = metrics.r2_score(y_test, Y_pred)
        #########################
        # # save model here
        # if rmse>threshold:
        #     with open('saved/knn.pickle', 'wb') as f:
        #         pickle.dump(neigh, f)
        #     threshold=rmse
        #########################

        #########################
        prediction_part.extend(Y_pred)
        index.extend(test_index)
        RMSE.append(rmse)
        RMSLE.append(rmsle)
        R2.append(r2)

    prediction_all = np.column_stack((index,prediction_part))

    return prediction_all, np.mean(RMSE), np.mean(RMSLE), np.mean(R2)


def LASSO(X, y, n_splits=5, random_state=666,
          alpha=0.01):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prediction_part = []
    index = []
    RMSE = []
    RMSLE = []
    R2 = []
    # threshold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,], y.iloc[test_index,]
        #########################
        # fit your model here
        lasso_raw=linear_model.Lasso(copy_X=True,normalize=True,precompute=True,alpha=alpha)
        lasso_raw.fit(X_train,y_train)
        Y_pred=lasso_raw.predict(X_test)
        # get metrics here
        rmse = np.sqrt(metrics.mean_squared_error(y_test,Y_pred))
        rmsle = np.sqrt(np.mean(np.square(abs(np.log1p(y_test) - np.log1p(Y_pred)))))
        r2 = metrics.r2_score(y_test, Y_pred)
        #########################
        # # save model here
        # if rmse>threshold:
        #     with open('saved/lasso.pickle', 'wb') as f:
        #         pickle.dump(lasso_raw, f)
        #     threshold=rmse
        #########################

        #########################
        prediction_part.extend(Y_pred)
        index.extend(test_index)
        RMSE.append(rmse)
        RMSLE.append(rmsle)
        R2.append(r2)

    prediction_all = np.column_stack((index,prediction_part))

    return prediction_all, np.mean(RMSE), np.mean(RMSLE), np.mean(R2)



def RIDGE(X, y, n_splits=5, random_state=666,
          alpha=0.01):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prediction_part = []
    index = []
    RMSE = []
    RMSLE = []
    R2 = []
    # threshold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,], y.iloc[test_index,]
        #########################
        # fit your model here
        ridge_raw=linear_model.Ridge(copy_X=True,normalize=True,solver='auto',alpha=alpha)
        ridge_raw.fit(X_train,y_train)
        Y_pred=ridge_raw.predict(X_test)
        # get metrics here
        rmse = np.sqrt(metrics.mean_squared_error(y_test,Y_pred))
        rmsle = np.sqrt(np.mean(np.square(abs(np.log1p(y_test) - np.log1p(Y_pred)))))
        r2 = metrics.r2_score(y_test, Y_pred)
        #########################
        # # save model here
        # if rmse>threshold:
        #     with open('saved/ridge.pickle', 'wb') as f:
        #         pickle.dump(ridge_raw, f)
        #     threshold=rmse
        #########################

        #########################
        prediction_part.extend(Y_pred)
        index.extend(test_index)
        RMSE.append(rmse)
        RMSLE.append(rmsle)
        R2.append(r2)

    prediction_all = np.column_stack((index,prediction_part))

    return prediction_all, np.mean(RMSE), np.mean(RMSLE), np.mean(R2)