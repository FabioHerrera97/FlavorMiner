import unittest
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
sys.path.insert(0, '../src/')

from molecule_manager import MoleculeManager
 
class TestMoleculeManager(unittest.TestCase):
 
    def setUp(self):

        self.data = pd.DataFrame({
            'smiles': ['CCO', 'CC(=O)O', 'C1=CC=CC=C1', 'invalid_smiles'],
            'property': [1, 2, 3, 4]})
        self.manager = MoleculeManager(data=self.data)
    
    def test_standardization_pipeline(self):

        result = self.manager.standardization_pipeline('smiles')

        self.assertEqual(len(result), 2)  # Two invalid SMILES should be removed ('CCO' and 'invalid_smiles')
        self.assertNotIn('invalid_smiles', result['smiles'].values)

 
    def test_create_mols_from_smiles(self):
        standardized_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1', '']
        self.manager.data['standardized_smiles'] = standardized_smiles
        result = self.manager.create_mols_from_smiles('standardized_smiles')
        expected_mols = [Chem.MolFromSmiles(smiles) for smiles in standardized_smiles]
        self.assertEqual(len(result), len(expected_mols))
        for res_mol, exp_mol in zip(result, expected_mols):
            if exp_mol is None:
                self.assertIsNone(res_mol)
            else:
                self.assertTrue(Chem.MolToSmiles(res_mol) == Chem.MolToSmiles(exp_mol))
 
    def test_create_3d_molecule(self):
        smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        result = self.manager.create_3d_molecule(mols)
        self.assertEqual(len(result), len(mols))
        for mol in result:
            self.assertIsNotNone(mol.GetConformer())

    def test_compute_rdkit_descriptors(self):
        mols = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC(=O)O')]
        df = self.manager.compute_rdkit_descriptors(mols)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)  # Two molecules
        self.assertGreater(len(df.columns), 0)
        
        for idx in df.index:
            self.assertFalse(df.loc[idx].isnull().all())
        
 
if __name__ == '__main__':
    unittest.main()
 