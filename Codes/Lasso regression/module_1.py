# Functions:
# 1. Get_Data(..) --- developed by Anri Karanovich
# 2. Get_Descriptors(..) --- developed by Nilotpal Chakraborty
# 3. Get_Fingerprints(..) --- developed by Nilotpal Chakraborty
# 4. Get_Input(..) --- developed by Anri Karanovich
# 5. Get_Output(..) --- developed by Anri Karanovich
# 6. LoadPreprocData(..) --- developed by Anri Karanovich
# 7. FileNames(..) --- developed by Anri Karanovich
# 8. DescriptorList() --- developed by Nilotpal Chakraborty
import os
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from pathlib import Path


# Get_Data(..) 

# Anri Karanovich, based on Nilotpal Chakraborty specific functions Get_Descriptors
# and Get_Fingerprints defined below


#   function reads the dataset from a file, 
#   based on the specified types of input data; process it 
#   if needed (normalizes, oh-encodes, or calculates descriptors
#   or fingerprints from to amend the input);
#   and store in a separate file if requested

# raw_data_file:  the name of the original txt dataset file (unimers or dimers)
#                  used only if source='txt'
# skiprows:  number of rows to skip when reading the raw data file. 
#                  used only if source='txt'
# subset:   subset == True - use only the first 1000 rows of the dataset
#           subset == False - use the whole dataset
# source:   source == 'pkl' - read preprocessed data from a pkl file
#           source == 'csv' - read preprocessed data from a csv file (default)
#           source == 'txt' - read raw data from a txt file and process it
# savepkl:  create the pkl file from the to store the processed data?

#   return X, Y  -  input and output arrays



def Get_Data(raw_data_file=None, skiprows=None, subset=False, 
                               source='csv', savepkl = False,
                               include_Inp = True, include_MD = False, 
                               include_FP = False, scaler = None):
    
    #1.1  Initialize parts of X as empty dataframes
    X_inp = pd.DataFrame()
    X_md = pd.DataFrame()
    X_fp = pd.DataFrame()
    
    
    #1.2  Get the necessary input/descriptors/fingerprints, normalize oif needed
    if include_Inp: 
        X_inp = Get_Input(raw_data_file=raw_data_file, skiprows=skiprows, 
                          subset=subset, source=source, savepkl = savepkl)
        if scaler is not None: X_inp[:]= scaler.fit_transform(X_inp)
    
    
    if include_MD: 
        X_md = Get_Descriptors(raw_data_file=raw_data_file, skiprows=skiprows, 
                               subset=subset, source=source, savepkl = savepkl)
        if scaler is not None: X_md[:]= scaler.fit_transform(X_md)
    
    
    if include_FP: 
        X_fp = Get_Fingerprints(raw_data_file=raw_data_file, skiprows=skiprows, 
                                      subset=subset, source=source, savepkl = savepkl)
        #fingerpints not normalized, as they are binary
    
    
    #1.3  Concatenate different portions of X
    X = pd.concat((X_inp, X_md, X_fp), axis=1)
        
    #1.4  Get output (Y)
    Y = Get_Output(raw_data_file=raw_data_file, skiprows=skiprows, 
                   subset=subset, source=source, savepkl = savepkl)
    
    return (X,Y)
    
    

# 2. Get_Descriptors(..) function, developed by Nilotpal Chakraborty
#   (if source=='csv', uses preprocessed data from Seokgyun Ham). 
#   Calulate the molecular descriptors
#   from the SMILES representations

def Get_Descriptors(raw_data_file=None, skiprows=None, subset=False, source='csv',
                    savepkl = True):
    
    # generate filenames
    
    filepkl, filecsv, fileraw = FileNames('MD', subset=subset,
                                          raw_data_file= raw_data_file)
    
    #Load preprocessed ddata, if it exists
    
    if (source == 'csv') or (source == 'pkl'):
        X = LoadPreprocData('MD', subset, source, savepkl)
               
    else:
        chosen_descriptors = DescriptorList()   # full descriptor list shown at the end
          	
        # create molecular descriptor calculator
        mol_descriptor_calculator = MolecularDescriptorCalculator(chosen_descriptors)
        
        # read the raw data file
        colnames = ["SMILES", "Martini", "DGA", "DGM", "apKa", "bpKa", "Acidity", "LogP"]  
               
        df = pd.read_csv(fileraw, sep=" ", skiprows=skiprows, names=colnames)
        df = df[df.Acidity != 'Z']     #drop zwitterionic compounds
        
        # calculation step
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES')
        ECFP6 = [mol_descriptor_calculator.CalcDescriptors(x) for x in df['ROMol']]
        #ecfp6_name = [f'MD_{i}' for i in range(200)]
        ecfp6_name = chosen_descriptors   
        ecfp6_bits = [list(l) for l in ECFP6]
        
        X = pd.DataFrame(ecfp6_bits, columns=ecfp6_name)
    
                  
    # save data to a pkl file of desired
    if savepkl == True:
        X.to_pickle(filepkl)
    
    return X



# 2. Get_Fingerprints(..) function, developed by Nilotpal Chakraborty
#   (if source=='csv', uses preprocessed data from Seokgyun Ham). 
#   Calulate the Morgan figerprint vectors
#   from the SMILES representations.

def Get_Fingerprints(raw_data_file=None, skiprows=None, subset=False, source='csv',
                    savepkl = False):
    
    # generate filenames
    
    filepkl, filecsv, fileraw = FileNames('FP', subset=subset,
                                          raw_data_file = raw_data_file)
    
    #Load preprocessed ddata, if it exists
    
    if (source == 'csv') or (source == 'pkl'):
        X = LoadPreprocData('FP', subset, source, savepkl)
        
    else:
        colnames = ["SMILES", "Martini", "DGA", "DGM", "apKa", "bpKa", "Acidity", "LogP"]  
                
        # read the raw data file
        df = pd.read_csv(fileraw, sep=" ",skiprows = skiprows, names=colnames)
        df = df[df.Acidity != 'Z']
            
        #perform claculations
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol='SMILES')
            
        radius = 2     #defines the Morgan fingerprint radius/order
        nBits = 1024   #Morgan fingerprint vector length
            
        ECFP6 = [AllChem.GetMorganFingerprintAsBitVect(x,radius=radius, nBits=nBits) for x in df['ROMol']]
        ecfp6_name = [f'bit_{i}' for i in range(nBits)]
        ecfp6_bits = [list(l) for l in ECFP6]
        	
        X = pd.DataFrame(ecfp6_bits, columns=ecfp6_name)
       
        #X = X.to_numpy()
        	#Y = y.to_numpy()
            
    # save data to a pkl file of desired
    if savepkl == True:
        X.to_pickle(filepkl)
            
    return X




# 4. Get_Input(..) function, written by Anri Karanovich.
#    (if source=='csv', uses preprocessed data from Seokgyun Ham). 
#    Read in the Column 3-7 features from the original dataset

def Get_Input(raw_data_file=None, skiprows=None, subset=False, source='csv',
                    savepkl = False):
    
    filepkl, filecsv, fileraw = FileNames('INPUT', subset=subset,
                                          raw_data_file=raw_data_file)
    
    #read proprocessed data
    if (source == 'csv') or (source == 'pkl'):
        
        X = LoadPreprocData('INPUT', subset, source, savepkl)
        
    else:
        colnames = ["SMILES", "Martini", "DGA", "DGM", "apKa", "bpKa", "Acidity", "LogP"]  
                
        # read the raw data file
        df = pd.read_csv(fileraw, sep=" ",skiprows = skiprows, names=colnames)
        df = df[df.Acidity != 'Z']

        X = df.drop(["SMILES","LogP","Martini"], axis=1)
        
        X['apKa'] = X['apKa'].replace('NOT', 20)        #replacins missing pKa
        X['bpKa'] = X['bpKa'].replace('NOT', -10)
        
        #one-hot encode Acidity
        
        X.insert(4,"Acidity_A",0)
        X.insert(5,"Acidity_B",0)
        X.insert(6,"Acidity_N",0)
        
        acid_a_list = X.index[df['Acidity']=='A'].tolist()
        acid_b_list = X.index[df['Acidity']=='B'].tolist()
        acid_n_list = X.index[df['Acidity']=='N'].tolist()
        
        X.loc[acid_a_list,'Acidity_A'] = 1
        X.loc[acid_b_list,'Acidity_B'] = 1
        X.loc[acid_n_list,'Acidity_N'] = 1
        
        X = X.drop('Acidity', axis=1)

    
    if savepkl == True:
        X.to_pickle(filepkl)
            
    return X




#5. Get Output  function. 
#   (if source=='csv', uses preprocessed data from Seokgyun Ham). 
#   Read the Log(P) values from the dataset

def Get_Output(raw_data_file=None, skiprows=None, subset=False, source='csv',
                    savepkl = False):
    
    filepkl, filecsv, fileraw = FileNames('OUTPUT', subset=subset,
                                          raw_data_file=raw_data_file)
    
    
    if (source == 'csv') or (source == 'pkl'):
        
        X = LoadPreprocData('OUTPUT', subset, source, savepkl)
    
    else:
        colnames = ["SMILES", "Martini", "DGA", "DGM", "apKa", "bpKa", "Acidity", "LogP"]  
                
        # read the raw data file
        df = pd.read_csv(fileraw, sep=" ",skiprows = skiprows, names=colnames)
        df = df[df.Acidity != 'Z']

        X = df['LogP']

    
    if savepkl == True:
        X.to_pickle(filepkl)
            
    return X

    

#6. LoadPreprocData(..).  Load the data (fingerprints, descriptors, 
#         column 2-7 inputs, or output) from a csv or a pkl file

def LoadPreprocData(file_prefix, subset=False, source='csv',
                    savepkl = False):
    
    filepkl, filecsv, fileraw = FileNames(file_prefix, subset=subset,
                                          raw_data_file=None)
        
    # AK: two ways tro load the preprocessed data - from cvs or pkl
    
    X = pd.DataFrame()
    
    if (source == 'pkl') and os.path.exists(filepkl):    #pkl fie for fast load
        X = pd.read_pickle(filepkl)
        
    elif (source == 'csv') and os.path.exists(filecsv):   #csv pre-processed file
        X = pd.read_csv(filecsv, index_col=False)
        if 'SMILES' in X.columns: X = X.drop("SMILES", axis=1)
        
    return X



#7. FileNames - return the names of the input files for given parameters

def FileNames(file_prefix, subset=False, raw_data_file=None):
    # AK: find the file with the data
    cwd = os.getcwd()
    slash = os.sep     
    filename = file_prefix + "_preprocessed_data"
    if subset==True: 
        postfix = '_1000'
    else:
        postfix=''
    
    prepath = cwd +slash +"Datasets" 
    filepkl = prepath +slash +filename +postfix +'.pkl'
    filecsv = prepath +slash +filename +postfix +'.csv'
    fileraw = ''
    if raw_data_file is not None: 
        fileraw = prepath + slash + raw_data_file
    
    filepkl = Path(filepkl)   #OS-dependent path format proided by Path function
    filecsv = Path(filecsv)
    fileraw = Path(fileraw)
    
    return filepkl, filecsv, fileraw
    
        
    
#8. DescriptorList - return the names of the molecular descriptors

def DescriptorList():
    chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', \
                          'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n',\
                          'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10',\
                          'EState_VSA11', 'EState_VSA2', 'EState_VSA3', \
                          'EState_VSA4', 'EState_VSA5', 'EState_VSA6', \
                          'EState_VSA7', 'EState_VSA8', 'EState_VSA9', \
                          'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',\
                          'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', \
                          'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', \
                          'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex',\
                          'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',\
                          'MinAbsEStateIndex', 'MinAbsPartialCharge',\
                          'MinEStateIndex', 'MinPartialCharge', 'MolLogP',\
                          'MolMR', 'MolWt', 'NHOHCount', 'NOCount', \
                          'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',\
                          'NumAliphaticRings', 'NumAromaticCarbocycles',\
                          'NumAromaticHeterocycles', 'NumAromaticRings', \
                          'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',\
                          'NumRadicalElectrons', 'NumRotatableBonds',\
                          'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',\
                          'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1',\
                          'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',\
                          'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', \
                          'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', \
                          'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', \
                          'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',\
                          'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',\
                          'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', \
                          'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',\
                          'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',\
                          'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10',\
                          'VSA_EState2', 'VSA_EState3', 'VSA_EState4',\
                          'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', \
                          'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert',\
                          'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', \
                          'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', \
                          'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',\
                          'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', \
                          'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate',\
                          'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',\
                          'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', \
                          'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',\
                          'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine',\
                          'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',\
                          'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',\
                          'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',\
                          'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',\
                          'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro',\
                          'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',\
                          'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', \
                          'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',\
                          'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', \
                          'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', \
                          'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',\
                          'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', \
                          'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane',\
                          'fr_urea', 'qed']
        
    return chosen_descriptors
