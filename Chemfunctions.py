import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors 
from sklearn.preprocessing import StandardScaler


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


def do_scaling(scaler=StandardScaler(), xtrain=None, xtest=None):
    """
    Usage: do_scaling(scaler=MinMaxScaler(), xtrain=xtrain, xtest=test) 
    xtrain and xtest are pd.Dataframes
    Caution: Do test_train_split before scaling
    Return: return scaled non-None xtrain and xtest
    """
    # from sklearn.preprocessing import StandardScaler
    st = scaler

    if xtrain is not None:
        col=xtrain.columns.values.tolist()
        xtrain=st.fit_transform(xtrain)  
        xtrain=pd.DataFrame(xtrain,columns=col)

        if xtest is not None:
            
            xtest=st.transform(xtest)
            xtest=pd.DataFrame(xtest,columns=col)
            print("returning scaled train and test data")
            return xtrain,xtest
        else:
            print("test data is not provided, returning only scaled train data")
            return xtrain
    else:
        print("Give train data, returning None")
        return xtrain,xtest

def check_normality_feat(df,alpha = 0.05):
    """
    This function will check the normality of each feature in the df
    Print out the list of non-normal and normal features 
    """
    from scipy.stats import shapiro
    print("Shapiro-Wilk test of normality")
    
    normal_feat= [False]*len(df.columns.to_list())
    w_stat_feat=np.zeros(len(df.columns.to_list()))
    p_value_feat=np.zeros(len(df.columns.to_list()))
    
    for idx,col in enumerate(df.columns.to_list()):            
        w_stat_feat[idx],p_value_feat[idx] = shapiro(df[col])
        
        if p_value_feat[idx] > alpha:
            normal_feat[idx]=True
        else:
            pass
    normal_feat_dict={}
    normal_feat_dict["Feature"]=df.columns.to_list()
    normal_feat_dict["Normal_dist"]=normal_feat
    normal_feat_dict["W_stat"]=w_stat_feat
    normal_feat_dict["p-value"]=p_value_feat
    
    df_nor=pd.DataFrame(normal_feat_dict)
    
    print(f"{df_nor['Normal_dist'].sum()} features are normally distributed")
    print(f'Returning a dataframe with {df_nor.columns.to_list()} columns')
    
    return df_nor