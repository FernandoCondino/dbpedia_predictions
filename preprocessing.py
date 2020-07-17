import pandas as pd
import bz2
import re
import csv
import os.path
from dbpedia_common import TARGET_HELD_OUT_CSV, TARGET_DEV_CSV, DBPEDIA_RAW_CSV, FINAL_DBPEDIA_RAW_CSV, \
    ALL_RELATIONS_CSV, parse_float, INPUT_FOLDER


def create_dbpedia_raw_df(random_state=23):
    """ Using geonames_cities file, we filter ALL db pedia body to select only the cities with +1000 population
        There are 3 files in db pedia to be parsed: instance_types, mapping_objects and mapping_literals
        If we already have a processed file, we skip all the processing and return it instead
    """
    dbpedia_service = DbPediaService()
    if not os.path.isfile(DBPEDIA_RAW_CSV):
        dbpedia_service.create_raw_df(DBPEDIA_RAW_CSV, random_state=random_state)

    return dbpedia_service.get_raw_dataframe(DBPEDIA_RAW_CSV)


# 57362 rows with population.
def extract_populations(dbpedia_raw_df):
    column_prefix = '<http://dbpedia.org/ontology/'
    population_verbs = ['populationTotal>', 'populationMetro>', 'populationUrban>', 'populationRural>']
    populations_df = dbpedia_raw_df[dbpedia_raw_df.relation.str.contains('|'.join(population_verbs))].copy()
    populations_df['object'] = populations_df['object'].apply(parse_float)

    pivot_df = populations_df.pivot_table(index='subject', columns='relation', values='object')
    pivot_df['target'] = pivot_df[f'{column_prefix}populationTotal>']
    pivot_df.loc[pivot_df.target.isnull(), 'target'] = pivot_df[f'{column_prefix}populationUrban>']
    pivot_df.loc[pivot_df.target.isnull(), 'estimated'] = pivot_df[f'{column_prefix}populationRural>'].fillna(0)

    print('Cities with populationTotal: ', (pivot_df[f'{column_prefix}populationTotal>'].notnull()).sum())
    print('Cities with populationUrban: ', (pivot_df[f'{column_prefix}populationUrban>'].notnull()).sum())
    print(f'Removed rows with less than 1000 population: {(pivot_df.target < 1000).sum()}')
    print(f'Rows with Missing population: {(pivot_df.target.isnull()).sum()}')

    pivot_df = pivot_df.drop(pivot_df[pivot_df.target < 1000].index)
    return pivot_df[['estimated', 'target']].copy()


def filter_raw_df(cols_to_remove, occurrence_threshold):
    target_df = pd.read_csv(TARGET_DEV_CSV).append(pd.read_csv(TARGET_HELD_OUT_CSV))
    dbpedia_raw_df = DbPediaService().get_raw_dataframe(DBPEDIA_RAW_CSV)
    merged = pd.merge(dbpedia_raw_df, target_df, on='subject', how='inner')

    print('BEFORE joining populations file and raw dbpedia file:')
    print(f'-- Unique subjects: {dbpedia_raw_df.subject.nunique()}')
    print(f'-- Size (rows): {len(dbpedia_raw_df)}')
    print('AFTER joining populations file and raw dbpedia file:')
    print(f'-- Unique subjects: {merged.subject.nunique()}')
    print(f'-- Size (rows): {len(merged)}')

    relations_df = (merged.groupby('relation').subject.nunique()
                    .rename('%_of_occurence')
                    .div(merged.subject.nunique())
                    .sort_values(ascending=False)
                    .reset_index())

    filter_mask = (relations_df['%_of_occurence'] < occurrence_threshold) | (
        relations_df.relation.str.contains('|'.join(cols_to_remove)))
    relations_df.loc[filter_mask, 'delete'] = True
    relations_df = relations_df.fillna(False)
    print(f'saving relations Dataframe in {ALL_RELATIONS_CSV}')
    relations_df.to_csv(ALL_RELATIONS_CSV, index=False)

    relations_to_keep = relations_df[relations_df.delete == False].relation.values
    final_raw_df = merged[merged.relation.isin(relations_to_keep)]

    final_raw_df.to_csv(FINAL_DBPEDIA_RAW_CSV, index=False)
    print(f'saving final_dbpedia_raw Dataframe in {FINAL_DBPEDIA_RAW_CSV}')
    print(f'AFTER removing rare relations with occurrences up to {occurrence_threshold * 100}%:')
    print(f'-- Unique subjects: {final_raw_df.subject.nunique()}')
    print(f'-- Unique relations: {final_raw_df.relation.nunique()}')
    return final_raw_df, relations_df


class DbPediaService:
    RAW_DATA_FOLDER = f'{INPUT_FOLDER}/raw_data'
    GEONAMES_CITIES_CSV = f'{RAW_DATA_FOLDER}/geonames-all-cities-with-a-population-1000.csv'
    GEONAMES_LINKS_TTL = f'{RAW_DATA_FOLDER}/geonames_links.ttl.bz2'
    INSTANCE_TYPES_TTL = f'{RAW_DATA_FOLDER}/instance-types_lang_en_transitive.ttl.bz2'
    MAPPING_LITERALS_TTL = f'{RAW_DATA_FOLDER}/mappingbased-literals_lang_en.ttl.bz2'
    INSTANCE_OBJECTS_TTL = f'{RAW_DATA_FOLDER}/mappingbased-objects_lang_en.ttl.bz2'

    def create_raw_df(self, dbpedia_raw_csv, random_state=23):
        """ Using geonames_cities file, we filter ALL db pedia body to select only cities with +1000 population
            There are 3 files in db pedia to be parsed: instance_types, mapping_objects and mapping_literals
            If we already have a processed file, we skip all the processing and return it instead
        """
        # We found 77599 different subjects (cities) in dbpedia
        geonames_city_ids = set(pd.read_csv(self.GEONAMES_CITIES_CSV, sep=';', header=0)['Geoname ID'].values)
        print(f'Cities found in Geonames project: {len(geonames_city_ids)}')
        dbpedia_service = DbPediaService()
        dbpedia_cities_set = dbpedia_service.get_dbpedia_cities_from_geonames(geonames_city_ids)
        print(f'Cities found in DBpedia: {len(dbpedia_cities_set)}')

        with open(dbpedia_raw_csv, 'w', encoding='utf-8') as file_handler:
            csv_writer = csv.writer(file_handler, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['subject', 'relation', 'object'])
            for line, subject, verb, obj in dbpedia_service.parse_db_pedia_bz2(self.INSTANCE_TYPES_TTL):
                if subject in dbpedia_cities_set:
                    csv_writer.writerow([subject, verb, obj])
            for line, subject, verb, obj in dbpedia_service.parse_db_pedia_bz2(self.MAPPING_LITERALS_TTL):
                if subject in dbpedia_cities_set:
                    csv_writer.writerow([subject, verb, obj])
            for line, subject, verb, obj in dbpedia_service.parse_db_pedia_bz2(self.INSTANCE_OBJECTS_TTL):
                if subject in dbpedia_cities_set:
                    csv_writer.writerow([subject, verb, obj])
                if obj in dbpedia_cities_set:
                    csv_writer.writerow([obj, f'{verb[:-1]}?inv>', {subject}])
        return dbpedia_raw_csv

    def get_dbpedia_cities_from_geonames(self, geonames_city_ids):
        city_id_mapping_regex = re.compile('sws\.geonames\.org/(\d+)/')
        dbpedia_cities = set([])
        with bz2.BZ2File(self.GEONAMES_LINKS_TTL, 'r') as db_pedia_mapping_file:
            for byteline in db_pedia_mapping_file:
                if byteline[0] == ord('#'):
                    continue
                line = byteline.decode('utf-8')
                regex_match = city_id_mapping_regex.search(line)
                if regex_match:
                    if int(regex_match.group(1)) in geonames_city_ids:
                        dbpedia_cities.add(line.split(' ')[0])
        return dbpedia_cities

    def parse_db_pedia_bz2(self, bz2_db_pedia_filename):
        db_pedia_regex = re.compile('(<.+?>)\s+(<.+?>)\s+(.+?) \.')
        with bz2.BZ2File(bz2_db_pedia_filename, 'r') as file_reader:
            for byteline in file_reader:
                if byteline[0] == ord('#'):
                    continue
                line = byteline.decode('utf-8')
                matcher = db_pedia_regex.match(line)
                if matcher:
                    subject, verb, obj = matcher.groups()
                    yield line, subject, verb, obj
                else:
                    print('********** did not match!:', line)

    def get_raw_dataframe(self, dbpedia_raw_csv):
        return pd.read_csv(dbpedia_raw_csv, sep=';', quotechar='|')


if __name__ == '__main__':
    pass
