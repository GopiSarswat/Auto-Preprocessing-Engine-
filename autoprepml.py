#Auto preprocessing pipeline for ML - handles loading, cleaning, encoding, scaling, and feature selection in one class.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from functools import wraps

def step_logger(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        print(f"\nRunning step: {func.__name__}")
        result = func(self, *args, **kwargs)
        print(f"Completed: {func.__name__}")
        return result
    return wrapper

class AutoCSVProcessor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    # generator for pipeline steps
    def pipeline(self):
        steps = [
            self.load_data,
            self.remove_constant_columns,
            self.remove_duplicates,
            self.handle_missing,
            self.handle_outliers,
            self.encode_categorical,
            self.scale_features,
            self.remove_multicollinearity,
            self.remove_low_variance
        ]

        for step in steps:
            yield step

    @step_logger
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("Shape:", self.df.shape)

    @step_logger
    def remove_constant_columns(self):
        self.df = self.df.loc[:, self.df.nunique() > 1]

    @step_logger
    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    @step_logger
    def handle_missing(self):
        num_cols = self.df.select_dtypes(include=['int64','float64']).columns
        cat_cols = self.df.select_dtypes(include=['object','category']).columns

        for col in num_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)

        for col in cat_cols:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    @step_logger
    def handle_outliers(self):
        num_cols = self.df.select_dtypes(include=['int64','float64']).columns

        for col in num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            self.df[col] = np.clip(self.df[col], lower, upper)

    @step_logger
    def encode_categorical(self):
        self.df = pd.get_dummies(self.df, drop_first=True)

    @step_logger
    def scale_features(self):
        scaler = StandardScaler()
        self.df[:] = scaler.fit_transform(self.df)

    @step_logger
    def remove_multicollinearity(self):
        corr_matrix = self.df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
        self.df.drop(columns=to_drop, inplace=True)

    @step_logger
    def remove_low_variance(self):
        selector = VarianceThreshold(threshold=0.01)
        self.df = pd.DataFrame(
            selector.fit_transform(self.df)
        )

    # single public function
    def process(self, save_path="model_ready_dataset.csv"):
        for step in self.pipeline():   # generator used here
            step()

        self.df.to_csv(save_path, index=False)

        print("\nFinal Shape:", self.df.shape)
        print("Saved to:", save_path)

        return self.df

processor = AutoCSVProcessor("dataset.csv")
model_ready_data = processor.process()