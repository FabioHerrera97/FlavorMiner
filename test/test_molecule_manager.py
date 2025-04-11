import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from unittest.mock import patch, MagicMock, create_autospec


import sys
sys.path.insert(0, '../src/')

from molecule_manager import MoleculeManager

try:
    from test_molecule_manager import MoleculeManager
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES, 
    reason='Required dependencies not valid'
)

TEST_SMILES = ['NC(Cl)(Br)C(=O)O', 'CCO', 'CC(=O)O', 'c1ccccc1', 'Invalid smiles']
TEST_DF = pd.DataFrame({'smiles': TEST_SMILES})

@pytest.fixture
def molecule_manager():
    ''' Fixture to create a MoleculeManager with test data'''
    manager = MoleculeManager(data=TEST_DF.copy())
    manager.standardization_pipeline = MagicMock(return_value=TEST_SMILES)
    return manager

def test_initialization():
    ''' Test initialization of MoleculeManagaer'''
    manager = MoleculeManager()
    assert manager.data is None

    manager_with_data = MoleculeManager(data=TEST_DF)
    assert manager_with_data.data.equals(TEST_DF)

def test_standardization_pipeline(mock_pipeline, molecule_manager):
    """Test standardization pipeline"""
    # Setup mock pipeline
    mock_instance = MagicMock()
    mock_instance.transform.return_value = TEST_SMILES
    mock_pipeline.return_value = mock_instance
    
    standardized = molecule_manager.standardization_pipeline('smiles')
    assert len(standardized) == len(TEST_SMILES)
    mock_pipeline.assert_called_once()

def test_create_mols_from_smiles(molecule_manager):
    """Test molecule creation from SMILES"""
    # Setup mock return values
    mock_mols = [MagicMock(spec=Chem.Mol) if s != "invalid_smiles" else None 
                for s in TEST_SMILES]
    
    with patch('rdkit.Chem.MolFromSmiles', side_effect=mock_mols):
        molecule_manager.data["standardized_smiles"] = TEST_SMILES
        mols = molecule_manager.create_mols_from_smiles("standardized_smiles")
        assert len(mols) == len(TEST_SMILES)
        # Check invalid SMILES returns None
        assert mols[-1] is None

def test_create_3d_molecule(molecule_manager):
    """Test 3D generation works with valid molecules"""
    mock_mol = MagicMock(spec=Chem.Mol)
    with patch('rdkit.Chem.AddHs', return_value=mock_mol), \
         patch('rdkit.Chem.AllChem.EmbedMolecule', return_value=0), \
         patch('rdkit.Chem.AllChem.UFFOptimizeMolecule', return_value=0):
        
        mols_3d = molecule_manager.create_3d_molecule([mock_mol])
        assert len(mols_3d) == 1
        assert mols_3d[0] == mock_mol

@patch('molecule_manager.FPVecTransformer')
def test_compute_ecfp(mock_calc, molecule_manager):
    """Test MACCS keys calculation"""
    mock_calc.return_value.return_value = [0]*1024
    mock_mol = MagicMock(spec=Chem.Mol)
    
    result = molecule_manager.compute_ecfp([mock_mol])
    assert isinstance(result, pd.DataFrame)


@patch('molecule_manager.FPVecTransformer')
def test_compute_maccs(mock_calc, molecule_manager):
    """Test MACCS computation returns dataframe"""
    mock_calc.return_value.return_value = [0]*166
    mock_mol = MagicMock(spec=Chem.Mol)
    
    result = molecule_manager.compute_maccs([mock_mol])
    assert isinstance(result, pd.DataFrame)

@patch('rdkit.ML.Descriptors.MoleculeDescriptors.MolecularDescriptorCalculator')
def test_compute_rdkit_descriptors(mock_calc, molecule_manager):
    """Test descriptor calculation returns dataframe"""
    mock_calc.return_value.GetDescriptorNames.return_value = ['MolWt', 'LogP']
    mock_calc.return_value.CalcDescriptors.return_value = [100, 2.5]
    mock_mol = MagicMock(spec=Chem.Mol)
    
    result = molecule_manager.compute_rdkit_descriptors([mock_mol])
    assert isinstance(result, pd.DataFrame)
    assert 'MolWt' in result.columns

@patch('molfeat.trans.pretrained.PretrainedMolTransformer')
@patch('molfeat.trans.pretrained.Pretrained.hf_transformers.PretrainedHFTTransformers')
def test_compute_graph_convolutional(molecule_manager):
    ''' Test graph convolutional embedding calculation'''
    standardized = molecule_manager.standardization_pipeline('smiles')
    
    results_df = molecule_manager.compute_graph_convolutional(standardized)

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == len(TEST_DF)
    assert not results_df.isnull().all().all()

def test_transformer_embeddings(mock_mol_transformer, mock_hf_transformer, molecule_manager):
    ''' Test all transformer-based embeddings methods with mocks'''
    standardized = molecule_manager.standardization_pipeline('smiles')
    molecule_manager.data['standardized_smiles'] = standardized

    mock_mol_transformer.return_value = MagicMock(return_value=np.random.rand(512))
    mock_hf_transformer.return_value = MagicMock(return_value=np.random.rand(768))

    # Test PCQM4Mv2 Graphormer
    graphormer_df = molecule_manager.compute_pcqm4mv2_graphormer(standardized)
    assert isinstance(graphormer_df, pd.DataFrame)
    assert len(graphormer_df) == len(TEST_DF)
    assert any(isinstance(col, int) for col in graphormer_df.columns) # Embedding dimensions

    # Test MolBERT
    molbert_df = molecule_manager.compute_molbert(standardized)
    assert isinstance(molbert_df, pd.DataFrame)
    assert len(molbert_df) == len(TEST_DF)
    assert any(isinstance(col, int) for col in molbert_df.columns)

    # Test MolFormer
    molformer_df = molecule_manager.compute_molformer(standardized)
    assert isinstance(molformer_df, pd.DataFrame)
    assert len(molformer_df) == len(TEST_DF)
    assert any(isinstance(col, int) for col in molformer_df.columns)

    # Test MolT5
    molt5_df = molecule_manager.compute_molt5(standardized)
    assert isinstance(molt5_df, pd.DataFrame)
    assert len(molt5_df) == len(TEST_DF)
    assert any(isinstance(col, int) for col in molt5_df.columns)

    # Test ChemBERTa-MTR
    chemberta_mtr_df = molecule_manager.compute_chemberta2_mtr(standardized)
    assert isinstance(chemberta_mtr_df, pd.DataFrame)
    assert len(chemberta_mtr_df) == len(TEST_DF)
    assert any(isinstance(col, int) for col in chemberta_mtr_df.columns)

    # Test ChemBERTA-MLM
    chemberta_mlm_df = molecule_manager.compute_chemberta2_mlm(standardized)
    assert isinstance(chemberta_mlm_df, pd.DataFrame)
    assert len(chemberta_mlm_df) == len(TEST_DF)
    assert any(isinstance(col, int) for col in chemberta_mlm_df.columns)

def test_error_handling():
    ''' Test error handling when no data is provided'''
    manager = MoleculeManager()

    with pytest.raises(AttributeError):
        manager.standardization_pipeline('smiles')

    with pytest.raises(AttributeError):
        manager.create_mols_from_smiles('smiles')