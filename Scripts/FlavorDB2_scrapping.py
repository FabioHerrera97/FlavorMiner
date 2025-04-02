import pandas as pd
from bs4 import BeautifulSoup
import urllib.request
import argparse

def flavordb2_entity_url(pubchem_id):
    '''Generates the FlavorDB2 URL for a given PubChem ID.

    Args:
        pubchem_id (str): The PubChem compound identifier.

    Returns:
        url (str): The complete URL to the FlavorDB2 molecule details page.
    '''

    url = f'https://cosylab.iiitd.edu.in/flavordb2/molecules_details?id={pubchem_id}'
    return url

def get_flavordb2_entity(pubchem_id):
    '''Retrieves aroma/taste threshold data from FlavorDB2 for a given PubChem ID.

    Args:
        pubchem_id (str): The PubChem compound identifier.

    Returns:
        str: The extracted threshold value or 'unknown' if not found.

    Note:
        Prints status messages about the retrieval process.
    '''

    url = flavordb2_entity_url(pubchem_id)

    with urllib.request.urlopen(url) as response:
        html = response.read()

    soup = BeautifulSoup(html, 'html.parser')

    panel = soup.find('div', {'id': f'{pubchem_id}_a_t_values'})

    if panel:
        
        list_items = panel.find_all('li', class_='list-group-item')

        for item in list_items:
            text = item.get_text(strip=True)

        print(f'✅ Found threshold: {text}')
    else:
        print('❌ No threshold data found (skipping)')

    return text

def data_extraction(data_file):
    '''Extracts and compiles flavor threshold data from FlavorDB2 for compounds in input file.

    Args:
        data_file (str): Path to CSV file containing compound data. Expected columns:
                         - pubChem ID
                         - Compound Name
                         - flavor profile
                         - Isomeric smiles
                         - Source

    Returns:
        list: A list of dictionaries containing compound data with added threshold values.
              Each dictionary contains:
              - Compound Name
              - pubChem ID
              - flavor profile
              - smiles
              - Source
              - Aroma/Taste Threshold Values

    Note:
        Prints progress information and handles errors for individual compounds.
    '''

    data = pd.read_csv(data_file)

    threshold_data = []

    for index, row in data.iterrows():
        pubchem_id = str(row['pubChem ID'])
        compound_name = row['Compound Name']
        flavor_profile = row['flavor profile']
        smiles = row['Isomeric smiles']
        source = row['Source']

        print (f'Processing: {compound_name} (PubChem ID: {pubchem_id})')

        try:
            threshold = get_flavordb2_entity(pubchem_id)
            if 'unknown' in threshold_data:
                pass
            else: 

                results = { 'Compound Name': compound_name, 
                            'pubChem ID': pubchem_id, 
                            'flavor profile': flavor_profile,
                            'smiles': smiles,
                            'Source':  source,
                            'Aroma/Taste Threshold Values': threshold}

            threshold_data.append(results)

        except Exception as e:
            print(f'⚠️ Error fetching data for {pubchem_id}: {e}')

    return threshold_data

def main():

    parser = argparse.ArgumentParser(description='Exracting threshold data from FlavorDB2')
    parser.add_argument('--input_file', type=str, help='excel file containing the data')

    args = parser.parse_args()

    data_extracted = data_extraction(args.input_file)
    df = pd.DataFrame(data_extracted)
    df.to_csv('../Data/threshold_data.csv', index=False)

if __name__== '__main__':
    main()