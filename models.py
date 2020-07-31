from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def get_linear_pipeline(alpha=1, countries_threshold=0.97, utc_threshold=0.95, log=False):
    preprocessing = Pipeline(steps=[
        ('countries', CategoricalThresholdTransformer('country#cat', threshold=countries_threshold, log=log)),
        ('utc_offset', CategoricalThresholdTransformer('utc_offset#cat', threshold=utc_threshold, log=log)),
        ('calculated_pop', CalculatedPopTransformer()),
    ])

    numeric_transformer = Pipeline(steps=[
        ('log', LogTransformer(exclude_columns=[])),
        ('poli', PolynomialFeatures(2)),
        ('scale', MinMaxScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    transformers = ColumnTransformer(
        transformers=[
            ('numeric_log', numeric_transformer, selector(dtype_exclude=['object', 'category'])),
            ('categorical', categorical_transformer, selector(dtype_include=['object', 'category'])),
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing),
        ('transformations', transformers),
        ('model', Ridge(alpha=alpha)),
    ])
    return pipeline


def get_svr_pipeline(countries_threshold=0.97, utc_threshold=0.95, log=False):
    preprocessing = Pipeline(steps=[
        ('countries', CategoricalThresholdTransformer('country#cat', threshold=countries_threshold, log=log)),
        ('utc_offset', CategoricalThresholdTransformer('utc_offset#cat', threshold=utc_threshold, log=log)),
        ('calculated_pop', CalculatedPopTransformer()),
    ])

    numeric_transformer = Pipeline(steps=[
        ('log', LogTransformer(exclude_columns=[])),
        ('scale', MinMaxScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    transformers = ColumnTransformer(
        transformers=[
            ('numeric_log', numeric_transformer, selector(dtype_exclude=['object', 'category'])),
            ('categorical', categorical_transformer, selector(dtype_include=['object', 'category'])),
        ], remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing),
        ('transformations', transformers),
        ('model', SVR(C=0.5, epsilon=0.01, gamma='scale', cache_size=1999)),
    ])
    return pipeline


def get_lxgb_pipeline():
    model = LGBMRegressor(
        learning_rate=0.02,
        n_estimators=3000,
        max_depth=7,
        feature_fraction=0.5,
        cat_smooth=1,
        bagging_freq=20,
        num_leaves=25,
        reg_alpha=0.8,
    )

    def set_as_category(X):
        df = X.copy()
        msk = df.dtypes == 'object'
        if msk.sum() > 0:
            df.loc[:, msk] = df.loc[:, msk].astype('category')
        return df

    def rename_cols(X):
        df = X.copy()
        df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
        return df

    rename_transformer = Pipeline(steps=[
        ('rename_columns', FunctionTransformer(rename_cols)),
    ])

    categorical_transformer = Pipeline(steps=[
        ('set_as_category', FunctionTransformer(set_as_category)),
    ])

    pipeline = Pipeline(steps=[
        ('calculated_pop', CalculatedPopTransformer()),
        ('categorical_transformer', categorical_transformer),
        ('rename_columns', rename_transformer),
        ('model', model),
    ])
    return pipeline


class ModelEvaluator:
    def __init__(self, model, X_test, y_test, X_train, y_train):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

    def evaluate_model(self, log=True, eval_r2_score=False):
        self.test_prediction = self.model.predict(self.X_test)
        self.train_prediction = self.model.predict(self.X_train)

        if log:
            print('*' * 20)
            print(f'Test RMSLE: {mean_squared_error(self.y_test, self.test_prediction) ** 0.5}')
            print(f'Test RMSE: {mean_squared_error(10 ** self.y_test, 10 ** self.test_prediction) ** 0.5}')
            if eval_r2_score:
                print(f'Test R2 score: {r2_score(self.y_test, self.test_prediction)}')
            print('*' * 20)
            print(f'Train RMSLE: {mean_squared_error(self.y_train, self.train_prediction) ** 0.5}')
            print(f'Train RMSE: {mean_squared_error(10 ** self.y_train, 10 ** self.train_prediction) ** 0.5}')
            if eval_r2_score:
                print(f'Test R2 score: {r2_score(self.y_train, self.train_prediction)}')
        return self.test_prediction, self.train_prediction

    def get_error_analysis(self, test_df):
        error_analysis_df = test_df.loc[self.y_test.index, ['subject', 'target']].copy()
        error_analysis_df['pretty_subject'] = error_analysis_df.subject.map(lambda subject: subject.split('/')[-1])
        error_analysis_df['predicted'] = 10 ** self.test_prediction
        error_analysis_df['log_predicted'] = self.test_prediction
        error_analysis_df['log_target'] = np.log10(error_analysis_df.target)
        error_analysis_df.loc[:, 'log_diff'] = error_analysis_df.log_predicted - error_analysis_df.log_target
        error_analysis_df.loc[:, 'diff'] = error_analysis_df.predicted - error_analysis_df.target
        error_analysis_df = (error_analysis_df.loc[error_analysis_df.log_diff.abs()
            .sort_values(ascending=False).index])
        error_analysis_df = error_analysis_df[['pretty_subject', 'log_diff', 'log_target', 'log_predicted', 'target',
                                               'predicted', 'diff', 'subject']]  # reordering columns
        return error_analysis_df

    def plot_results(self, outlier_limit=1.5):
        fig, ax = plt.subplots(3, 1, figsize=(20, 12))
        good_mask = np.abs(self.test_prediction - self.y_test) < outlier_limit
        bad_mask = np.abs(self.test_prediction - self.y_test) >= outlier_limit

        sns.regplot(x=self.y_test[bad_mask], y=self.test_prediction[bad_mask] - self.y_test[bad_mask], ax=ax[0],
                    color='red', fit_reg=False)
        sns.regplot(x=self.y_test[good_mask], y=self.test_prediction[good_mask] - self.y_test[good_mask], ax=ax[0],
                    color='seagreen', fit_reg=False)
        ax[0].axhline(0, ls='--', color='black')
        ax[0].set(xlabel='Target', ylabel='Prediction')
        ax[0].set_title('Residual Plot')

        sns.regplot(x=self.y_test[bad_mask], y=self.test_prediction[bad_mask], ax=ax[1], color='red', fit_reg=False)
        sns.regplot(x=self.y_test[good_mask], y=self.test_prediction[good_mask], ax=ax[1], color='royalblue',
                    fit_reg=False)
        ax[1].plot([3, 7], [3, 7], 'red', linewidth=2)
        ax[1].set_title('Predicted vs Target scatter plot')

        ax[2].plot(self.test_prediction, label="log predicted", color='gray')
        ax[2].plot(self.y_test.values, label="log target", color='black')
        ax[2].set_title('Predicted (gray) and Target (black)')
        plt.tight_layout()


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.drop(columns=self.exclude_columns) if self.exclude_columns else X.copy()
        # ignoring this for now..
        elevation_column = '<http://dbpedia.org/ontology/elevation>'
        if elevation_column in df.columns.values:
            df.loc[df[elevation_column] < 0, elevation_column] = 0
        df = np.log1p(df)
        if self.exclude_columns:
            df[self.exclude_columns] = X[self.exclude_columns].copy()
        return df


class CategoricalThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, threshold, log=True):
        self.threshold = threshold
        self.column = column
        self.log = log

    def fit(self, X, y=None):
        df = X.copy()
        value_counts = df[self.column].value_counts()
        included_mask = (value_counts.cumsum() / value_counts.sum()) <= self.threshold
        self.included_values = value_counts.index[included_mask].tolist()
        if self.log:
            print(
                f'Selecting {len(self.included_values)} {self.column} of {len(value_counts)} to get {self.threshold * 100}% of the data')
            print(f'Renaming the rest of the {self.column} values to OTHER')
        return self

    def transform(self, X):
        df = X.copy()
        df.loc[~df[self.column].isin(self.included_values), self.column] = 'OTHER'
        return df


class CalculatedPopTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pop_density = '<http://dbpedia.org/ontology/populationDensity>'
        self.area_land = '<http://dbpedia.org/ontology/areaLand>'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mask = (X[self.pop_density] != 0) & (X[self.area_land] != 0)
        area = X.loc[mask, self.area_land]
        density = X.loc[mask, self.pop_density]

        copy_x = X.copy()
        copy_x['calc_population'] = area.mul(density / 1_000_000)
        copy_x.loc[copy_x['calc_population'] < 1000, 'calc_population'] = np.nan  # Fixing bad data.
        copy_x['calc_population_NAN'] = copy_x['calc_population'].isnull().astype(int)
        copy_x['calc_population'] = copy_x['calc_population'].fillna(0)
        return copy_x
