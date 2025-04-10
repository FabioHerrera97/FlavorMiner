import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from molfeat.cal import FPCalculator, DescriptorCalculator
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained import PretrainedMolTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTTransformer
from molfeat.trans.pretrained import GraphormerTransformer
from deepchem.feat import MolGraphFeaturizer
from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2mol import (ElementFilter, MetalDisconnector, SaltRemover, StereoRemover, SolventRemover, 
                                 TautomerCanonicalizer, Uncarger)

class MoleculeManager:

    '''
    A class for managing and processing molecular data, including standardization, featurization, and descriptor calculation.

    The MoleculeManager provides a comprehensive toolkit for handling molecular representations,
    including SMILES standardization, molecular fingerprint calculation, 2D/3D descriptor computation,
    and transformer-based embeddings from various pretrained models.

    Attributes:
        data (pd.DataFrame): Input data containing molecular information.
        maccs_calculator (FPCalculator): Calculator for MACCS fingerprints.
        graph_featurizer (MolGraphFeaturizer): Featurizer for graph convolutional features.
        molbert_transformer (PretrainedMolTransformer): Transformer for MolBERT embeddings.
        molformer_transformer (PretrainedMolTransformer): Transformer for MolFormer embeddings.
        molt5_transformer (PretrainedHFTTransformer): Transformer for MolT5 embeddings.
        chemberta_2_mtr_transformer (PretrainedHFTTransformer): Transformer for ChemBERTa-MTR embeddings.
        chemberta_2_mlm_transformer (PretrainedHFTTransformer): Transformer for ChemBERTa-MLM embeddings.

    Methods:
        standardization_pipeline(smiles_col): Standardizes SMILES strings through multiple chemical transformations.
        create_mols_from_smiles(stand_smiles_col): Converts standardized SMILES to RDKit molecule objects.
        create_3d_molecule(mols): Generates 3D conformations for molecules.
        compute_ecfp(mols, radius, nBits): Computes ECFP fingerprints for molecules.
        compute_maccs(mols): Computes MACCS fingerprints for molecules.
        compute_rdkit_descriptors(mols): Calculates RDKit 2D molecular descriptors.
        compute_3d_descriptors(mols_3D): Computes 3D molecular descriptors.
        compute_graph_convolutional(smiles_list): Generates graph convolutional features.
        compute_molbert(smiles_list): Computes MolBERT embeddings.
        compute_molformer(smiles_list): Computes MolFormer embeddings.
        compute_molt5(smiles_list): Computes MolT5 embeddings.
        compute_chemberta2_mtr(smiles_list): Computes ChemBERTa-MTR embeddings.
        compute_chemberta2_mlm(smiles_list): Computes ChemBERTa-MLM embeddings.

    '''

    def __init__(self, data=None):

        '''
        Initializes the DataPreprocessor with either a DataFrame.

        Args:
            data (pd.DataFrame, optional): Direct DataFrame input. Defaults to None.
        '''
        
        self.data = data
        self.maccs_calculator = FPCalculator('maccs')
        self.graph_featurizer = MolGraphFeaturizer()
        self.molbert_transformer = None
        self.molformer_transformer = None
        self.molt5_transformer = None
        self.chemberta_2_mtr_transformer = None
        self.chemberta_2_mlm_transformer = None

    def standardization_pipeline(self, smiles_col):

        '''
        Applies a standardized chemical transformation pipeline to SMILES strings.

        Performs a series of chemical standardizations including:
        - Conversion to molecule objects
        - Metal disconnection
        - Solvent removal
        - Salt removal
        - Element filtering
        - Uncharging
        - Tautomer canonicalization
        - Stereo information removal

        Args:
            smiles_col (str): Name of the column containing SMILES strings to standardize

        Returns:
            list: Standardized SMILES strings after all transformations
        '''

        pipeline_standardization = Pipeline([
            ('auto2mol', AutoToMol()),
            ('Metal_disconnector', MetalDisconnector()),
            ('Solvent_remover', SolventRemover()),
            ('Salt_remover', SaltRemover()),
            ('Element_remover', ElementFilter()),
            ('Uncharge', Uncarger()),
            ('Stereo_remover', StereoRemover()),
            ('Canonical_tautomer', TautomerCanonicalizer())
        ])

        standardized_smiles = pipeline_standardization.transform(self.data[smiles_col])

        return standardized_smiles
    
    def create_mols_from_smiles(self, stand_smiles_col):
        '''
        Converts standardized SMILES strings to RDKit molecule objects.

        Args:
            stand_smiles_col (str): Name of the column containing standardized SMILES strings

        Returns:
            list: List of RDKit Mol objects. Molecules that fail to parse will be None.

        Note:
            The input SMILES should ideally be pre-standardized using standardization_pipeline()
        '''

        smiles_list = self.data[stand_smiles_col]
        mols = [Chem.MolFromSmiles(i) for i in smiles_list]
        
        return mols
    
    def create_3d_molecule(self, mols):

        '''
        Generates 3D conformations for molecules using RDKit's distance geometry.

        For each molecule:
        1. Adds hydrogens
        2. Generates a 3D conformation using distance geometry
        3. Optimizes the geometry using UFF forcefield (For more accurate results, consider using MMFF94 or other force fields)

        Args:
            mols (list): List of RDKit Mol objects (2D molecules)

        Returns:
            list: List of RDKit Mol objects with 3D coordinates

        Note:
            This operation can be time-consuming for large numbers of molecules.
            Input molecules should have proper valence and sanitization.
        '''

        def mol_to_3d(mol):

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)

            return mol
        
        molecules_3d = [mol_to_3d(mol) for mol in mols]

        return molecules_3d
    
    def compute_ecfp(self, mols, radius, nBits=1024):

        '''
        Computes Extended Connectivity FingerPrints (ECFP) for a list of molecules.

        Generates circular Morgan fingerprints as bit vectors of specified size.

        Args:
            mols (list): List of RDKit Mol objects
            radius (int): The radius of the circular fingerprint (typically 2-3)
                        Higher radius captures larger molecular neighborhoods
            nBits (int, optional): Length of the fingerprint bit vector. Defaults to 1024.

        Returns:
            pd.DataFrame: Original dataframe concatenated with ECFP features.
                        Features are named 'Bit_0' to 'Bit_{nBits-1}'

        Note:
            - ECFP4 (radius=2) is the most commonly used variant
            - Larger nBits reduces collision probability but increases dimensionality
        '''

        ecfp = [AllChem.GetMorganFingerprintAsBitVector(mol, radius=radius, nBits=nBits) for mol in mols]
        ecfp_name = [f'Bit_{i}' for i in range(nBits)]
        ecfp_bits = [list(x) for x in ecfp]
        df_ecfp = pd.DataFrame(ecfp_bits, index= self.data.index, columns= ecfp_name)
        df_all = pd.concat([self.data, df_ecfp], axis=1)

        return df_all
    
    def compute_maccs(self, mols):

        '''
        Computes MACCS (Molecular ACCess System) keys for a list of molecules.

        Generates the 166-bit MACCS structural keys fingerprint that encodes
        specific substructural features and pharmacophoric patterns.

        Args:
            mols (list): List of RDKit Mol objects

        Returns:
            pd.DataFrame: Original dataframe concatenated with MACCS keys.
                        Column names represent specific structural features.

        Note:
            - MACCS keys are predefined structural patterns
            - Useful for coarse similarity screening
            - Less flexible but more interpretable than ECFP
        '''
       
        maccs = [self.maccs_calculator(i) for i in mols]
        df_maccs = pd.DataFrame(maccs, index=self.data.index)
        df_all = pd.concat([self.data, df_maccs], axis=1)

        return df_all

    def compute_rdkit_descriptors(self, mols):

        '''
        Computes comprehensive set of 2D molecular descriptors using RDKit.

        Calculates ~200 molecular descriptors including:
        - Physical properties (MW, logP, etc.)
        - Topological indices
        - Electronic features
        - Pharmacophoric properties

        Args:
            mols (list): List of RDKit Mol objects

        Returns:
            pd.DataFrame: Dataframe containing all computed RDKit descriptors.
                        Column names correspond to descriptor names.

        Note:
            - Some descriptors may be NaN for certain molecules
            - Descriptors should be standardized before machine learning
            - Includes both simple and complex descriptors
        '''

        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        desc_names = calc.GetDescriptorNames()

        mol_descriptors = []

        for mol in mols:
            descriptors = calc.CalcDescriptors(mol)
            mol_descriptors.append(descriptors)

        descriptor_df = pd.DataFrame(mol_descriptors, columns=desc_names)

        return descriptor_df
    
    def compute_3d_descriptors(self, mols_3D):

        '''
        Computes 3D molecular descriptors from molecules with 3D coordinates.

        Generates descriptors that capture spatial and conformational features
        including:
        - PMI (Principal Moments of Inertia)
        - Radial distribution functions
        - Surface area descriptors
        - Conformational statistics

        Args:
            mols_3D (list): List of RDKit Mol objects with 3D coordinates

        Returns:
            pd.DataFrame: Original dataframe concatenated with 3D descriptors

        Note:
            - Molecules must have 3D coordinates (use create_3d_molecule first)
            - Descriptors are sensitive to conformation quality
            - Some descriptors may be NaN for small or problematic molecules
        '''
        
        transformer = FPVecTransformer(kind='desc3D', dtype=float)
        features = transformer(mols_3D)
        df_3d = pd.DataFrame(features, index=self.data.index)
        df_all = pd.concat([self.data, df_3d], axis=1)
        
        return df_all
    
    def compute_graph_convolutional(self, smiles_list):

        '''
        Computes graph convolutional features for molecules from SMILES strings.

        Generates learned molecular representations by applying graph neural network
        operations on the molecular graph structure.

        Args:
            smiles_list (list): List of standardized SMILES strings

        Returns:
            pd.DataFrame: Original dataframe concatenated with graph features

        Note:
            - Uses pretrained graph featurizer
            - Features capture topological and chemical patterns
            - Returns None for molecules that fail featurization
            - Features are typically low-dimensional compared to fingerprints
        '''
        
        def featurize_smiles(smiles):
            try:
                features = self.graph_featurizer.featurize([smiles])[0]
                return features
            except:
                return None
            
        graph_features = [featurize_smiles(i) for i in smiles_list]
        df_graph_features = pd.DataFrame(graph_features, index=self.data.index)
        df_all = pd.concat([self.data, df_graph_features], axis=1)

        return df_all
    
    def compute_pcqm4mv2_graphormer(self, smiles_list):
        
        ''' 
        Computes molecular embeddings using the PCQM4Mv2 Graphormer pretrained model.

        The Graphormer model is a graph transformer architecture that has been pretrained
        on the PCQM4Mv2 quantum chemistry dataset. It generates embeddings by:
        1. Converting SMILES to molecular graphs
        2. Applying graph attention with spatial encoding
        3. Producing fixed-size vector representations

        Args:
            smiles_list (list): List of valid SMILES strings to process. Should be
                pre-standardized for optimal results.

        Returns:
            pd.DataFrame: Original dataframe concatenated with Graphormer embeddings.
                Each column represents a dimension of the embedding vector (typically
                768 dimensions for the base model).

        Raises:
            ValueError: If input SMILES list is empty or contains only invalid entries
            RuntimeError: If model initialization fails (e.g., missing dependencies)

        Note:
            - Uses the base version of PCQM4Mv2 Graphormer (128M parameters)
            - Embeddings capture both structural and quantum chemical properties
            - For large datasets, consider batching the SMILES processing

        '''

        transformer = GraphormerTransformer(kind='pcqm4mv2_graphormer_base', dtype=float)
        features = [transformer(smiles) for smiles in smiles_list]

        df_graphormer = pd.DataFrame(features, index=self.data.index)
        df_all = pd.concat([self.data, df_graphormer], axis=1)

        return df_all
    
    def compute_molbert(self, smiles_list):

        '''
        Generates molecular embeddings using the MolBERT pretrained transformer model.

        MolBERT is a bidirectional transformer model pretrained on molecular SMILES strings
        using masked language modeling.

        Args:
            smiles_list (list): List of standardized SMILES strings to process

        Returns:
            pd.DataFrame: Original dataframe concatenated with MolBERT embeddings.
                        Each column represents a dimension of the embedding vector.

        Note:
            - First call will initialize the MolBERT transformer (may take time/memory)
            - Input SMILES should be pre-standardized for best results
            - Embedding dimensionality is 512
        '''

        if self.molbert_transformer is None:
            self.molbert_transformer = PretrainedMolTransformer('molbert', dtype=np.float32)
        molbert_vectors = [self.molbert_transformer(i) for i in smiles_list]
        df_molbert = pd.DataFrame(molbert_vectors, index=self.data.index)
        df_all = pd.concat([self.data, df_molbert], axis=1)

        return df_all
    
    def compute_molformer(self, smiles_list):

        '''
        Computes molecular embeddings using the MolFormer pretrained transformer model.

        MolFormer is a large-scale molecular transformer trained on extensive chemical data,
        capable of generating context-aware molecular representations. It uses rotary position
        embeddings for improved chemical pattern recognition.

        Args:
            smiles_list (list): List of standardized SMILES strings

        Returns:
            pd.DataFrame: Original dataframe concatenated with MolFormer embeddings.
                        Columns represent dimensions of the embedding space.

        Note:
            - Initializes the transformer on first call (memory intensive)
            - Produces higher-dimensional embeddings than MolBERT (typically 768-1280 dim)
            - Suitable for both small molecules and larger chemical structures
            - Embeddings capture both local and global molecular features
        '''

        if self.molformer_transformer is None:
            self.molformer_transformer = PretrainedMolTransformer('molformer', dtype=np.float32)
        molformer_vectors = [self.molformer_transformer(i) for i in smiles_list]
        df_molformer = pd.DataFrame(molformer_vectors, index=self.data.index)
        df_all = pd.concat([self.data, df_molformer], axis=1)

        return df_all
    
    def compute_molt5(self, smiles_list):

        '''
        Generates molecular embeddings using the MolT5 pretrained transformer model.

        MolT5 is a molecular transformer based on Google's T5 architecture, trained
        for both molecular property prediction and SMILES translation tasks. It produces
        embeddings particularly useful for cross-modal tasks.

        Args:
            smiles_list (list): List of valid SMILES strings

        Returns:
            pd.DataFrame: Original dataframe concatenated with MolT5 embeddings.
                        Embedding dimensions are model-dependent.

        Note:
            - Uses HuggingFace transformer implementation
            - Embeddings optimized for sequence-to-sequence chemical tasks
            - Supports both canonical and isomeric SMILES
            - May require SMILES standardization for optimal performance
        '''

        if self.molt5_transformer is None:
            self.molt5_transformer = PretrainedHFTTransformer(kind='molT5', notation='smiles', dtype=float)
        molt5_vectors = [self.molt5_transformer(i) for i in smiles_list]
        df_molt5 = pd.DataFrame(molt5_vectors, index=self.data.index)
        df_all = pd.concat([self.data, df_molt5], axis=1)

        return df_all
    
    def compute_chemberta2_mtr(self, smiles_list):

        '''
        Computes molecular embeddings using ChemBERTa-2 (MTR variant) pretrained model.

        The MTR (Multi-Task Regression) variant of ChemBERTa-2 is optimized for
        molecular property prediction tasks. It produces embeddings tuned for
        quantitative structure-property relationship modeling.

        Args:
            smiles_list (list): List of standardized SMILES strings

        Returns:
            pd.DataFrame: Original dataframe concatenated with ChemBERTa-2 MTR embeddings.
                        Column names correspond to embedding dimensions.

        Note:
            - Specifically fine-tuned for regression tasks
            - 77 million parameter model
            - Embeddings capture features relevant to physicochemical properties
            - Requires SMILES input (not other molecular representations)
        '''

        if self.hemberta_2_mtr_transformer is None:
            self.chemberta_2_mtr_transformer = PretrainedHFTTransformer(kind='ChemBERTa-77M-MTR', notation='smiles', dtype=float)
        chemberta2_mtr_vectors = [self.chemberta_2_mtr_transformer(i) for i in smiles_list]
        df_chemberta2_mtr = pd.DataFrame(chemberta2_mtr_vectors, index=self.data.index)
        df_all = pd.concat([self.data, df_chemberta2_mtr], axis=1)

        return df_all
    
    def compute_chemberta2_mlm(self, smiles_list):

        '''
        Generates molecular embeddings using ChemBERTa-2 (MLM variant) pretrained model.

        The MLM (Masked Language Modeling) variant of ChemBERTa-2 is trained using
        traditional language model objectives on chemical SMILES strings. It produces
        general-purpose molecular embeddings suitable for diverse downstream tasks.

        Args:
            smiles_list (list): List of valid SMILES strings

        Returns:
            pd.DataFrame: Original dataframe concatenated with ChemBERTa-2 MLM embeddings.
                        Each column represents an embedding dimension.

        Note:
            - Base model trained with masked token prediction
            - 77 million parameter architecture
            - Generates context-sensitive molecular representations
            - Embeddings may require task-specific fine-tuning for optimal performance
            - Handles SMILES syntax variations robustly

        '''
        if self.chemberta_2_mtr_transformer is None:
            self.chemberta_2_mtr_transformer = PretrainedHFTTransformer(kind='ChemBERTa-77M-MTR', notation='smiles', dtype=float)
        chemberta2_mtr_vectors = [self.chemberta_2_mtr_transformer(i) for i in smiles_list]
        df_chemberta2_mtr = pd.DataFrame(chemberta2_mtr_vectors, index=self.data.index)
        df_all = pd.concat([self.data, df_chemberta2_mtr], axis=1)

        return df_all