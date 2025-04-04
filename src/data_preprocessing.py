import pandas as pd
import requests
import re

class DataPreprocessor:

    '''
    A class for preprocessing dataframes with flavor threshold data.

    This class provides methods for cleaning data, calculating mean values from ranges,
    converting units to ppm, and fetching SMILES notations from compound names.

    Methods:
        clean_unknow_threshold: Removes rows containing 'unknown' values in a specified column.
        calculate_mean_from_range: Calculates mean values from range strings (e.g., "1-5" becomes 3).
        convert_units_ppm: Converts various concentration units to parts per million (ppm).
        get_smiles_from_name: Fetches SMILES notations for compounds using PubChem's API.
        get_threshold_info: Extracts threshold values and units from text data in the specified column.
    '''
    def __init__(self, data=None):
        '''
        Initializes the DataPreprocessor with either a DataFrame.

        Args:
            data (pd.DataFrame, optional): Direct DataFrame input. Defaults to None.
        '''

        self.data = data

    def clean_unknow_threshold(self, threshold_column):
        '''
        Removes rows containing 'unknown' values in the specified threshold column.

        Args:
            threshold_column (str): Name of the column to check for 'unknown' values.

        Returns:
            pandas.DataFrame: A new DataFrame with rows containing 'unknown' values removed.
        '''

        new_data = self.data[~self.data[threshold_column].str.contains('unknown')]
        return new_data
    
    def calculate_mean_from_range(self, threshold_col):

        '''
        Processes a column containing numeric ranges and calculates their mean values.

        Args:
            threshold_col (str): Name of the column containing range strings (e.g., "1-5").

        Returns:
            pandas.DataFrame: The input DataFrame with an additional 'processed_threshold' column
                            containing the calculated mean values.
        '''

        def process_values(value):
            if '-' in value:
                start, end = map(float, value.split('-'))
                return (start + end) / 2
            else:
                return float(value)
            
        self.data['processed_threshold'] = self.data[threshold_col].apply(process_values)

        return self.data
        

    def convert_units_ppm(self, threshold_column, units_column):

        '''
        Converts various concentration units to parts per million (ppm).

        Supported units: ppm, µg/kg, ng/g, µg/L, ppb, mg/m3, mg/kg.

        Args:
            threshold_column (str): Name of the column containing concentration values.
            units_column (str): Name of the column containing unit strings.

        Returns:
            pandas.DataFrame: The input DataFrame with an additional 'Threshold_ppm' column
                            containing the converted values.
        '''

        conversion_factor = {'ppm': 1, 'µg/kg': 1, 'ng/g': 0.001, 'µg/L': 1, 'ppb': 0.001, 'mg/m3': 1000, 'mg/kg': 1000}

        self.data['Threshold_ppm'] = self.data.apply(lambda row: row[threshold_column] * conversion_factor[row[units_column]], axis=1)

        return self.data
    
    def get_smiles_from_name(self, name_col):

        '''
        Fetches SMILES notations for compounds using PubChem's REST API.

        Args:
            name_col (str): Name of the column containing compound names.

        Returns:
            pandas.DataFrame: The input DataFrame with an additional 'smiles' column
                            containing the fetched SMILES strings. Compounds not found
                            or with errors will have corresponding messages in this column.

        Note:
            This method makes web requests to PubChem and may take time depending on
            the number of compounds and network speed. It prints the status of each
            lookup to the console.
        '''

        compound_names = self.data[name_col]
        smiles_list = []

        for i in compound_names:

            i = i.replace(' ', '')
            try:
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{i}/property/CanonicalSMILES/TXT'
                smiles = requests.get(url).text.rstrip()
                smiles_list.append(smiles)

                if ('NotFound' in smiles):
                    print(i, ': Not found ❌ ')
                else:
                    print(i, ': smiles found ✅')
            except:
                smiles = 'Problem with the url'
                smiles_list.append(smiles)
                print(i, ': There was an error in the url of this compound ⚠️')

        self.data['smiles'] = smiles_list
        return self.data
    
    def get_threshold_info(self, threshold_col):
        '''
        Extracts threshold values and units from text data in the specified column.

        Parses text in the given column to identify detection thresholds or levels, extracting both
        the numerical value/range and the units of measurement. The results are stored in two new
        columns ('Threshold_value' and 'Units') in the DataFrame.

        Args:
            threshold_col (str): Name of the column containing threshold information text to parse.

        Returns:
            pd.DataFrame: The modified DataFrame with two new columns:
                - 'Threshold_value': Contains extracted threshold values (may be single values or ranges)
                - 'Units': Contains the units of measurement (e.g., 'ppb', 'µg', 'ppm')

        Notes:
            - The function looks for patterns like "detection level 5-10 ppb" or "threshold at 0.5ppm"
            - Value ranges can be expressed with "to", "-", or commas (e.g., "1 to 5", "1-5", "1,000-5,000")
            - Supported units include: ppb, ppm, µg, ng, pb (with various combinations)
            - If no threshold is found, both new columns will contain None for that row
        '''
        def extract_threshold_info(text):

            pattern = r'''
                (?:detection\s*(?:level|threshold)?|at|as|is)\s*  # keywords before value
                ([\d.,]+\s*(?:to|-)\s*[\d.,]+)                    # Value or range
                \s*([µm]?[gpn]b|ppm|ppb)                          # Units
                '''
            match = re.search(pattern, text, re.IGNORECASE | re.VERBOSE)

            if match:
                value_range = match.group(1).strip()
                units = match.group(2).strip()

                value_range = value_range.replace('to', '-').replace(',', '')
                value_range = re.sub(r'\s*-\*', '-', value_range)

                return pd.Series([value_range, units])
            
            return pd.Series([None, None])
        
        self.data[['Threshold_value', 'Units']] = self.data[threshold_col].apply(extract_threshold_info)

        return self.data