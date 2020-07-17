import pandas as pd
import seaborn as sns
import os
from IPython.display import display_html

sns.set_palette('husl')
sns.set(style='whitegrid')
pd.options.display.max_columns = None
pd.set_option('display.float_format', lambda x: '%.3f' % x)

DF_FOLDER = 'dataframes'
INPUT_FOLDER = 'input'
TARGET_HELD_OUT_CSV = f'{INPUT_FOLDER}/target_held_out.csv'
TARGET_DEV_CSV = f'{INPUT_FOLDER}/target_dev.csv'
DBPEDIA_RAW_CSV = f'{INPUT_FOLDER}/intermediate_dbpedia_raw.csv'
FINAL_DBPEDIA_RAW_CSV = f'{INPUT_FOLDER}/final_dbpedia_raw.csv'
ALL_RELATIONS_CSV = f'{INPUT_FOLDER}/all_relations.csv'


def parse_float(text):
    if text.startswith('"'):
        text = text[1: text[1:].index('"') + 1]
    return float(text)


def print_df(df):
    display_html(df.to_html(), raw=True)


def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)


class DataFrameRepository:
    def __init__(self, version):
        self.version = version

    def get(self, test=False):
        path = self._get_path(test)
        return pd.read_csv(path) if os.path.isfile(path) else None

    def save(self, df, test=False):
        df.to_csv(self._get_path(test), index=False)

    def _get_path(self, test):
        suffix = 'held_out' if test else 'dev'
        return f'{DF_FOLDER}/{self.version}_{suffix}.csv'
