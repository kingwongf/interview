import pandas as pd
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
from tools.featGen import get_norm_side, moskowitz_func, tanh_func, mrm_c
from scipy.stats.mstats import winsorize
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', -1)  # or 199



class xgb(object):
    def __init__(self, data, test_date_split, target, model_loc, imp_feat_loc, min_ret_flt, side=True, grid_search=False, model_params=None):

        self.Xy = data.astype('float64')
        self.target = target
        self.test_date_split = test_date_split
        self.side = side
        self.model_loc = model_loc
        self.imp_feat_loc = imp_feat_loc
        self.min_ret_flt = min_ret_flt
        self.grid_search = grid_search
        self.model_params = model_params
        self.model = None
        self.numeric_features = None
        # print(self.Xy.describe())

    def min_ret(self, target_col, threshold):
        # target_col = winsorize(target_col, [0.05, 0.05])
        target_col.loc[target_col > threshold] = 1
        target_col.loc[target_col< -threshold] = -1
        target_col.loc[(target_col >= -threshold) &((target_col <= threshold))] = 0
        return target_col

    def set_target(self, col):
        if self.side:
            # self.Xy['target'] = get_norm_side(self.Xy[col], (self.Xy["emaret1m"], self.Xy["retvol1m"], 1.645))
            # self.Xy['target'] = np.sign(self.Xy[col])
            self.Xy['target'] = self.min_ret(self.Xy[col], self.min_ret_flt)
            # self.Xy['target'] = self.Xy['target'].astype('category')
            # print(self.Xy['target'])
        # self.Xy['target'] = to_categorical(get_norm_side(self.Xy[col], (self.Xy["emaret"], self.Xy["retvol1m"], 1.645)).astype('category'),3)
        # self.Xy = self.Xy[self.Xy["target"].notnull()]
        else:
            # self.Xy['target'] = self.Xy[col]
            self.Xy['target'] = tanh_func(winsorize(self.Xy[col], [0.05, 0.05]))
            # self.Xy['target'] = moskowitz_func(self.Xy[col])
            # self.Xy['target'] = tanh_func(self.Xy[col])
            # self.Xy['target'] = mrm_c(self.Xy[col])

        self.Xy = self.Xy.drop([col], axis=1)

        # print("percentage of nulls")
        # print(self.Xy.isnull().mean())

    def train_test_split(self):
        return self.Xy[:self.test_date_split], self.Xy[self.test_date_split:]

    @staticmethod
    def Xy_split(df, label):
        df1 = df.copy()
        target = df1.pop(label)
        return df1, target
    '''
    def gen_model(self):
        clf_params = {'num_class': 3, 'objective': 'multi:softprob', 'max_depth': 8,
                      'n_estimators': 300}  # , 'n_estimators':100
        reg_params = {'max_depth': 10, 'objective': 'reg:squarederror', 'n_estimators': 300}

        return XGBClassifier(**clf_params) if self.side else XGBRegressor(**reg_params)
    '''

    def gen_feature(self):
        categorical_features = self.Xy.columns[
            (self.Xy.dtypes.values != np.dtype('float64')) & (self.Xy.columns != 'target')].tolist()
        numeric_features = self.Xy.columns[
            (self.Xy.dtypes.values == np.dtype('float64')) & (self.Xy.columns != 'target')].tolist()
        return categorical_features, numeric_features

    def make_pipeline(self):
        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='median')),
                   ('scalar', StandardScaler())
                   ]
        )

        _, numeric_features = self.gen_feature()

        self.numeric_features = numeric_features

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ]
        )

        if self.model_params is None:
            clf_params = {'num_class': 3, 'objective': 'multi:softprob', 'max_depth': 8, 'n_estimators': 300, 'learning_rate':0.01}  # , 'n_estimators':100
            reg_params = {'max_depth': 10, 'objective': 'reg:squarederror', 'n_estimators': 300}
        else:
            clf_params = {**self.model_params, **{'num_class': 3, 'objective': 'multi:softprob', 'learning_rate':0.01} }
            reg_params = {**self.model_params, **{'objective': 'reg:squarederror', 'learning_rate':0.01} }

        under_sampler = RandomUnderSampler(random_state=42)
        if self.side:
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('undersampler', under_sampler),
                                  ('classifier', XGBClassifier(**clf_params))])
            self.model = clf
            return clf
        else:
            reg = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', XGBRegressor(**reg_params))])

            return reg

    def imp_feat(self):
        feature_importance = pd.Series(data=self.model.named_steps['classifier'].feature_importances_,
                                       index=self.numeric_features)

        return feature_importance.sort_values(ascending=False)

    def fit_model(self):
        print("Preparing to train...")
        # print(train_X.shape)
        self.set_target(self.target)

        clf = self.make_pipeline()

        train, test = self.train_test_split()
        # print(train, test)
        train_X, train_y = self.Xy_split(train,'target')
        test_X, test_y = self.Xy_split(test,'target')

         # print(sorted(clf.get_params().keys()))


        if self.grid_search:
            params_dict = {'classifier__max_depth': [2, 4, 6, 8],
                            'classifier__n_estimators': [50, 100, 200, 300, 500]}
            model_search = GridSearchCV(clf, params_dict, scoring='balanced_accuracy', verbose=1)
            model_search.fit(train_X, train_y)
            print(f"Best parameter (CV score={model_search.best_score_:.2f}")
            print(model_search.best_params_)
            return model_search.best_params_, model_search.best_score_

        else:
            clf.fit(train_X, train_y)

            pred_y = clf.predict(test_X)

            self.model = clf

            full_X, full_y =  self.Xy_split(self.Xy, 'target')
            clf.fit(full_X, full_y)


            ## save model
            pickle.dump(clf, open(self.model_loc, 'wb'))

            ## save important feature list
            imp_feat = self.imp_feat()
            imp_feat.rename(index={'long_y1_short_y2_target':'lagged_ret_long_y1_short_y2'})
            imp_feat.reset_index().to_csv(self.imp_feat_loc)

            # print("Test Accuracy", accuracy_score(test_y, pred_y))

            print("Test result: ")
            if self.side:
                print(classification_report(test_y, pred_y))
                print(confusion_matrix(test_y, pred_y, labels=[-1, 0, 1]))
            else:
                # print(np.concatenate([test_y,pred_y], axis=1))
                print(r2_score(test_y, pred_y))


