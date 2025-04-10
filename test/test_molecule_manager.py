import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from unittest.mock import patch, MagicMock


import sys
sys.path.insert(0, '../src/')

from molecule_manager import MoleculeManager

TEST_SMILES = ['NC(Cl)(Br)C(=O)O', 'CCO', 'CC(=O)O', 'c1ccccc1', 'Invalid smiles']
TEST_DF = pd.DataFrame({'smiles': TEST_SMILES})

@pytest.fixture
def molecule_manager():
    ''' Fixture to create a MoleculeManager with test data'''
    return MoleculeManager(data=TEST_DF)

def test_initialization():
    ''' Test initialization of MoleculeManagaer'''
    manager = MoleculeManager()
    assert manager.data is None

    manager_with_data = MoleculeManager(data=TEST_DF)
    assert manager_with_data.data.equals(TEST_DF)
    assert manager_with_data.data.maccs_calculator is not None
    assert manager_with_data.data.graph_featurizer is not None

def test_standardization_pipeline(molecule_manager):
    ''' Test SMILES standardization pipeline'''
    standardized = molecule_manager.standadization_pipeline('smiles')

    assert len(standardized) == len(TEST_SMILES)
    assert all(isinstance(s, str) or s is None for s in standardized)
    assert standardized[-1] is None # Check that invalid smiles returns None

def test_create_mols_from_smiles(molecule_manager):
    ''' Test molecule creation from smiles'''

    standardized = molecule_manager.standardization_pipeline('smiles') # standardize the smiles before mol creation
    molecule_manager.data['standardize_smiles'] = standardized

    mols = molecule_manager.create_mols_from_smiles('standardized_smiles')

    assert len(mols) == len(TEST_SMILES)
    assert all(isinstance(mol, Chem.Mol) or mol is None for mol in mols)
    assert mols[-1] is None # Invalid SMILES should result in None

def test_create_3d_molecule(molecule_manager):
    ''' Test 3D molecule generation'''
    standardized = molecule_manager.standardization_pipeline('smiles')
    molecule_manager.data['standardized_smiles'] = standardized
    mols = molecule_manager.create_mols_from_smiles('standardized_smiles')

    valid_mols = [mol for mol in mols if mol is not None]
    mols_3d = molecule_manager.create_3d_molecule(valid_mols)

    assert len(mols_3d) == len(valid_mols)
    for mol in mols_3d:
        assert mol.GetNumConformers() == 1 # Should have one conformer
        assert mol.GetNumAtoms() > 0

def test_compute_ecfp(molecule_manager):
    '''Test ECFP fingerprint calculation'''
    standardized = molecule_manager.standardization_pipeline('smiles')
    molecule_manager.data['standadized_smiles'] = standardized
    mols = molecule_manager.create_mols_from_smiles('standardized_smiles')

    valid_mols = [mol for mol in mols if mol is not None]
    radius = 2
    nBits = 1024

    result_df = molecule_manager.compute_ecfp(valid_mols, radius, nBits)

    assert isinstance(result_df, pd.DataFrame)
    assert f'Bit_{nBits-1}' in result_df.columns
    assert len(result_df) == len(TEST_DF)
    assert 'smiles' in result_df.columns

def test_compute_maccs(molecule_manager):
    ''' Test maccs fingerprint calculation'''
    standardized = molecule_manager.standardization_pipeline('smiles')
    molecule_manager.data['standardized_smiles'] = standardized
    mols = molecule_manager.create_mols_from_smiles('standardized_smiles')

    valid_mols = [mol for mol in mols if mol is not None]

    result_df = molecule_manager.compute_maccs(valid_mols)

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == len(TEST_DF)
    assert any(col.startwith('MACCS') for col in result_df.columns)

def test_compute_rdkit_descriptors(molecule_manager):
    ''' Test RDKit molecular descriptor calculations'''
    standardized = molecule_manager.standardization_pipeline('smiles')
    molecule_manager.data['standardized_smiles'] = standardized
    mols = molecule_manager.create_mols_from_smiles('standardized_smiles')

    valid_mols = [mol for mol in mols if mol is not None]

    results_df = molecule_manager.compute_rdkit_descriptors(valid_mols)

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df.columns) > 100
    assert 'MolWt' in results_df.columns # Check for a common descriptor

@patch('molfeat.trans.pretrained.PretrainedMolTransformer')
def test_compute_graph_convolutional(molecule_manager):
    ''' Test graph convolutional embedding calculation'''
    standardized = molecule_manager.standardization_pipeline('smiles')
    
    results_df = molecule_manager.compute_graph_convolutional(standardized)

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == len(TEST_DF)
    assert not results_df.isnull().all().all()