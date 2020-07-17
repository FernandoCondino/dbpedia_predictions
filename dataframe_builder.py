import pandas as pd
import numpy as np
from dbpedia_common import TARGET_HELD_OUT_CSV, TARGET_DEV_CSV, DBPEDIA_RAW_CSV

SELECTED_TYPE_PLACES = ['<http://www.wikidata.org/entity/Q3957>', '<http://www.wikidata.org/entity/Q532>',
                        '<http://schema.org/City>', '<http://dbpedia.org/ontology/Region>',
                        '<http://www.wikidata.org/entity/Q23442>', '<http://schema.org/Country>']


class DbpediaTidyDataframeBuilder:
    def __init__(self, raw_dbpedia_df, train_set=True):
        self.raw_dbpedia_df = raw_dbpedia_df
        target_csv = TARGET_DEV_CSV if train_set else TARGET_HELD_OUT_CSV
        self.final_dbpedia_df = pd.read_csv(target_csv)

    def with_numeric_columns(self, numeric_columns):
        raw_df = self.raw_dbpedia_df.copy()
        numeric_mask = raw_df.relation.isin(numeric_columns)
        raw_df.loc[numeric_mask, 'object'] = raw_df.loc[numeric_mask, 'object'].apply(parse_float)
        pivot_numeric_df = raw_df.loc[numeric_mask].pivot_table(index='subject', columns='relation', values='object',
                                                                aggfunc='max')
        pivot_numeric_df = pivot_numeric_df.fillna(0).astype(float).reset_index()

        dbpedia_df = self.final_dbpedia_df.merge(pivot_numeric_df, on='subject', how='left')
        for column in numeric_columns:
            cleaned_column = column.replace('<http://dbpedia.org/ontology/', '') + 'NAN'  # adding flag column for NaN's
            dbpedia_df.loc[:, cleaned_column] = dbpedia_df[column].isnull().astype(int)
        self.final_dbpedia_df = dbpedia_df.fillna(0)
        return self

    def with_counter_columns(self):
        pivot_counts = self.raw_dbpedia_df.pivot_table(index='subject', columns='relation', values='object',
                                                       aggfunc='count')
        pivot_counts = pivot_counts.fillna(0).astype(int)
        pivot_counts.columns = [col.split('/')[-1][:-1] + '#count' for col in pivot_counts.columns]
        self.final_dbpedia_df = self.final_dbpedia_df.merge(pivot_counts.reset_index(), on='subject', how='left')
        return self

    def with_rare_relations_count(self, relations_df):
        rare_relations = relations_df[relations_df.delete == True].relation # getting relations marked as rare
        dbpedia_raw_df = pd.read_csv(DBPEDIA_RAW_CSV, sep=';', quotechar='|') # using original df that has ALL relations
        rare_df = dbpedia_raw_df[dbpedia_raw_df.relation.isin(rare_relations.values)]
        rare_pivot = rare_df.pivot_table(index='subject', columns='relation', aggfunc=len).fillna(0)
        rare_pivot = rare_pivot.sum(axis=1).astype(int).rename('rare_rel#count').to_frame().reset_index()
        self.final_dbpedia_df = self.final_dbpedia_df.merge(rare_pivot, on='subject', how='left')
        return self

    def with_total_relations_count(self):
        pivot = self.raw_dbpedia_df.pivot_table(index='subject', columns='relation', aggfunc='count')
        pivot.columns = pivot.columns.droplevel(0)
        pivot_df = pivot.sum(axis=1).rename('total_rel#count').astype(int).to_frame().reset_index()
        self.final_dbpedia_df = self.final_dbpedia_df.merge(pivot_df, on='subject', how='left')
        return self

    def with_unique_relations_count(self):
        pivot = self.raw_dbpedia_df.pivot_table(index='subject', columns='relation', aggfunc='count')
        pivot.columns = pivot.columns.droplevel(0)
        pivot_df = pivot.count(axis=1).rename('unique_rel#count').astype(int).to_frame().reset_index()
        self.final_dbpedia_df = self.final_dbpedia_df.merge(pivot_df, on='subject', how='left')
        return self

    def with_place_types(self):
        column = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'

        place_type_df = self.raw_dbpedia_df[self.raw_dbpedia_df.relation == column].copy()
        place_type_df = place_type_df.loc[place_type_df.object.isin(SELECTED_TYPE_PLACES)]

        self._clean_place_type_names(place_type_df)
        place_type_df = place_type_df[['subject', 'object']].rename(columns={'object': 'place_type#cat'})

        d = self.raw_dbpedia_df[self.raw_dbpedia_df.relation == column].pivot_table(index='subject', columns='object',
                                                                                    values='relation', aggfunc='count')
        assert (d[SELECTED_TYPE_PLACES].sum(axis=1).value_counts().index.max() == 1)  # asserting a subject has only one place

        self.final_dbpedia_df = pd.merge(self.final_dbpedia_df, place_type_df, on='subject', how='left')
        self.final_dbpedia_df.loc[:, 'place_type#cat'] = self.final_dbpedia_df['place_type#cat'].fillna('NAN')
        return self

    def _clean_place_type_names(self, place_type_df):
        place_type_df.loc[place_type_df['object'].str.endswith('Q3957>'), 'object'] = 'Town'
        place_type_df.loc[place_type_df['object'].str.endswith('Q532>'), 'object'] = 'Village'
        place_type_df.loc[place_type_df['object'].str.endswith('City>'), 'object'] = 'City'
        place_type_df.loc[place_type_df['object'].str.endswith('Region>'), 'object'] = 'Region'
        place_type_df.loc[place_type_df['object'].str.endswith('Q23442>'), 'object'] = 'Island'
        place_type_df.loc[place_type_df['object'].str.endswith('Country>'), 'object'] = 'Country'

    def with_offset_types(self):
        column = '<http://dbpedia.org/ontology/utcOffset>'
        new_name = 'utc_offset#cat'
        utc_df = (self.raw_dbpedia_df[self.raw_dbpedia_df.relation == column]
                  .groupby('subject').object.apply(','.join).rename(new_name).reset_index())
        utc_df = self._clean_offset_values(utc_df, new_name)
        self.final_dbpedia_df = pd.merge(self.final_dbpedia_df, utc_df, on='subject', how='left')
        self.final_dbpedia_df['utc_offset#cat'] = self.final_dbpedia_df['utc_offset#cat'].fillna('NAN')
        return self

    def _clean_offset_values(self, df, column):
        strings_to_remove = '|'.join(['UTC', '±', '"",', '"', ' ', '\+', 'GMT', ':00'])
        df[column] = df[column].str.replace(strings_to_remove, '')
        df[column] = df[column].str.replace('&minus;', '-')
        df[column] = df[column].str.replace('−', '-')
        df[column] = df[column].str.replace('–', '-')
        df[column] = df[column].str.replace('-?0(\d)', lambda m: f'-{m.group(1)}')
        df[column] = df[column].replace('None', np.NaN)
        return df

    def with_countries(self):
        column = '<http://dbpedia.org/ontology/country>'
        new_name = 'country#cat'
        countries_df = (self.raw_dbpedia_df[self.raw_dbpedia_df.relation == column]
                        .copy().set_index('subject')
                        .object.rename(new_name).to_frame())

        countries_df['country_frequency'] = countries_df[new_name].map(countries_df[new_name].value_counts())
        countries_df = (countries_df.reset_index()
            .sort_values(['subject', 'country_frequency'])[~countries_df.index.duplicated(keep='last')])
        assert (countries_df.index.duplicated().sum() == 0)  # asserting there are no subjects with two countries
        countries_df = countries_df.drop(columns=['country_frequency'])

        self._clean_country_names(countries_df, new_name)
        self.final_dbpedia_df = self.final_dbpedia_df.merge(countries_df, on='subject', how='left')
        self.final_dbpedia_df.loc[:, new_name] = self.final_dbpedia_df[new_name].fillna('NAN')
        return self

    def _clean_country_names(self, countries_df, new_name):
        countries_df.loc[:, new_name] = countries_df[new_name].map(lambda name: name.split('/')[-1][:-1])
        countries_df.loc[countries_df[new_name].str.contains('China'), new_name] = 'China'
        countries_df.loc[countries_df[new_name].str.contains('Congo'), new_name] = 'Congo'

    def build(self):
        return self.final_dbpedia_df


def parse_float(text):
    if text.startswith('"'):
        text = text[1: text[1:].index('"') + 1]
    return float(text)
