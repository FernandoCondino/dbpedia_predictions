import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re


def compare_population_with_web(error_analysis_df, target_col='target', limit=30):
    if limit:
        error_analysis_df = error_analysis_df.loc[error_analysis_df.log_diff.abs().sort_values(ascending=False).index]
    else:
        limit = 300  # just in case
    population_data_list = []
    for subject_uri, expected_value in error_analysis_df[['subject', target_col]].values[:limit]:
        population_data_list.append(PopulationData(subject_uri, expected_value))

    analysis_df = make_analysis(population_data_list)
    error_subjects = analysis_df.loc[analysis_df.error == True, 'pretty_subject'].values
    print(
        f'{len(error_subjects)} Errors found while parsing wikipedia webpage. Make a manual check for: {error_subjects}')
    suspicious_subjects = analysis_df.loc[analysis_df.relative_log_diff.abs() > 0.5, 'pretty_subject'].values
    print(
        f'There are {len(suspicious_subjects)} suspicious subjects! Check if found values are real and fix the records')
    return analysis_df


def get_population_row(population_table):
    tags_dict = {i: tag for i, tag in
                 enumerate(population_table.find_all('tr', {'class': ['mergedtoprow', 'mergedrow']}))}
    population_index = -1
    for i, tag in tags_dict.items():
        if 'mergedtoprow' in tag.get('class'):
            if tag.find('th') and 'population' in tag.find('th').text.lower():
                population_index = i + 1
                break
    if population_index >= 0:
        index = population_index if tags_dict.get(population_index, False) else population_index - 1
        return tags_dict[index].find('td')
    return None


def analyse(subject_uri, expected):
    city_name = subject_uri.split('/')[-1][:-1]
    wikipedia_url = 'https://en.wikipedia.org/wiki/' + requests.utils.quote(city_name)
    html_page = requests.get(wikipedia_url)

    soup = BeautifulSoup(html_page.content, features='lxml')
    population_table = soup.find('table', {'class': 'geography'})
    data = {'subject': subject_uri, 'pretty_subject': city_name, 'expected_value': expected,
            'wiki_link': wikipedia_url, 'found_value': None, 'error': True, 'message': ''}
    if population_table:
        population_row = get_population_row(population_table)
        if population_row:
            text = str(population_row)
            matcher = re.search('^<td>([\d,]+).+', text)
            if matcher:
                found_population = int(matcher.group(1).replace(',', ''))
                data['found_value'] = int(found_population)
                if found_population > 950:
                    data['error'] = False
                else:
                    data['message'] = 'Too small population'
            else:
                data['message'] = f'Failed to get population from: {text}'
        else:
            data['message'] = f'Not found population population row for {subject_uri}'
    else:
        data['message'] = f'Not found population table for {subject_uri}'
    return data


class PopulationData:
    def __init__(self, subject_uri, expected_value):
        self.subject_uri = subject_uri
        self.expected_value = expected_value


def make_analysis(population_data_list):
    data = []
    for population_data in population_data_list:
        data.append(analyse(population_data.subject_uri, population_data.expected_value))
    analysis_df = pd.DataFrame(data)
    values = analysis_df[['expected_value', 'found_value']]
    analysis_df['relative_log_diff'] = np.log10((values.min(axis=1) / values.max(axis=1)).abs()).abs()
    analysis_df = analysis_df[
        ['pretty_subject', 'error', 'expected_value', 'found_value', 'relative_log_diff', 'message', 'wiki_link',
         'subject']]
    return analysis_df.sort_values(['error', 'relative_log_diff'], ascending=[False, False])
