import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors 
pd.set_option('display.max_columns', None)

def RDkit_descriptors(smiles):
    """
    Function will return all 208 RDKit descriptors
    smiles is a pandas series or a list of smiles
    """
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem import Descriptors
    mols = [Chem.MolFromSmiles(i) for i in smiles] 
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:
        # add hydrogens to molecules
        # mol=Chem.AddHs(mol)
        # Calculate all 208 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names 

def remove_nan_columns(df):
    columns_with_nan=df.columns[df.isna().any()]    
    df = df.dropna(axis='columns')
    print(f"Removed {len(columns_with_nan)} columns with nan")
    return df

def remove_duplicate_columns(df):
    
    print(f"Removed {sum(df.columns.duplicated())} duplicate columns")
    df=df.loc[:,~df.columns.duplicated()].copy()
    return df

def remove_columns_uniquevalues(df):
    print(f"Removed {sum(df.nunique()<2)} columns values with a unique value")
    df=df.loc[:,df.nunique()>1]
    return df

def remove_columns_low_std(df,threshold=0.3):
    print(f"Removed {sum(df.std() < threshold)} columns with std < {threshold} ")
    df=df.loc[:, df.std() >= threshold]
    return df

def remove_corr_features(df,corr_cutoff = 0.75):
    """
    This function will drop highly correlated features in the df
    Output: df without correlated features
    """
    cor_matrix=df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_cutoff)]
    print(f"Dropped {len(to_drop)} features with correlation coeff. > {corr_cutoff:0.2f}")

    df=df.drop(columns=to_drop,axis=1)
    return df

def remove_duplicate_smiles(df,smi="SMILES"):
    df[smi]=df[smi].apply(lambda x:Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    print(f"Removed {sum(df.duplicated(subset=smi))} duplicate SMILES from the given df")
    df.drop_duplicates(subset=[smi])
    return df