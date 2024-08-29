import torch
from typing import Optional, Any, Union, Callable, Tuple, List
import os
import glob
import numpy as np
import re
from tqdm import tqdm
from pdb import set_trace



# funtion to read the data
def read_data_dirs(
        dirs_names : List[str] = ['../intracardiac_dataset/data_hearts_dd_0p2/'],
        verbose : int = 0) -> List[List[str]]:
    """
    Read the numpy files in the given directories.
    Returns a list of file pairs ECG/Vm.
    
    Parameters
    ----------
    dirs_names : List[str]
        List of directories containing the data.
    verbose : int
        Verbosity level.
    
    Returns
    -------
    file_pairs : List[List[str]]
        List of file pairs.
    """
    file_pairs = []
    
    for dir in dirs_names:    
        all_files = sorted(glob.glob(dir + '/*.npy'))
        files_Vm=[]
        files_pECG=[]
        
        if verbose > 0:
            print('Reading files...',end='')
        for file in all_files:
            if 'VmData' in file:
                files_Vm.append(file)
            if 'pECGData' in file:
                files_pECG.append(file)
        if verbose > 0:        
            print(' done.')
        
        if verbose > 0:
            print('len(files_pECG) : {}'.format(len(files_pECG)))
            print('len(files_Vm) : {}'.format(len(files_Vm)))
        
        for i in range(len(files_pECG)):  
            VmName =  files_Vm[i]
            VmName = VmName.replace('VmData', '')
            pECGName =  files_pECG[i]
            pECGName = pECGName.replace('pECGData', '')            
            if pECGName == VmName :
                file_pairs.append([files_pECG[i], files_Vm[i]])
            else:
                print('Automatic sorted not matching, looking for pairs ...',end='')
                for j in range(len(files_Vm)):
                    VmName =  files_Vm[j]
                    VmName = VmName.replace('VmData', '')
                    if pECGName == VmName :
                        file_pairs.append([files_pECG[i], files_Vm[j]])
                print('done.')       
    return file_pairs


# function to transform the data
def get_standard_leads(
        pECGnumpy : np.ndarray
    ) -> np.ndarray :
    """
    Get the standard 12-lead from the 10-lead ECG.
    
    Parameters
    ----------
    pECGnumpy : np.ndarray
        10-lead ECG.
        
    Returns
    -------
    ecg12aux : np.ndarray
        12-lead ECG.
    """
    # pECGnumpy  : RA LA LL RL V1 V2 V3 V4 V5 V6
    # ecg12aux : i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6
    ecg12aux = np.zeros((pECGnumpy.shape[0],12))
    WilsonLead = 0.33333333 * (pECGnumpy[:,0] + pECGnumpy[:,1] + pECGnumpy[:,2])
    # Lead I: LA - RA
    ecg12aux[:,0] = pECGnumpy[:,1] - pECGnumpy[:,0]
    # Lead II: LL - RA
    ecg12aux[:,1] = pECGnumpy[:,2] - pECGnumpy[:,0]
    # Lead III: LL - LA
    ecg12aux[:,2] = pECGnumpy[:,2] - pECGnumpy[:,1]
    # Lead aVR: 3/2 (RA - Vw)
    ecg12aux[:,3] = 1.5*(pECGnumpy[:,0] - WilsonLead)
    # Lead aVL: 3/2 (LA - Vw)
    ecg12aux[:,4] = 1.5*(pECGnumpy[:,1] - WilsonLead)
    # Lead aVF: 3/2 (LL - Vw)
    ecg12aux[:,5] = 1.5*(pECGnumpy[:,2] - WilsonLead)
    # Lead V1: V1 - Vw
    ecg12aux[:,6] = pECGnumpy[:,4] - WilsonLead
    # Lead V2: V2 - Vw
    ecg12aux[:,7] = pECGnumpy[:,5] - WilsonLead
    # Lead V3: V3 - Vw
    ecg12aux[:,8] = pECGnumpy[:,6] - WilsonLead
    # Lead V4: V4 - Vw
    ecg12aux[:,9] = pECGnumpy[:,7] - WilsonLead
    # Lead V5: V5 - Vw
    ecg12aux[:,10] = pECGnumpy[:,8] - WilsonLead
    # Lead V6: V6 - Vw
    ecg12aux[:,11] = pECGnumpy[:,9] - WilsonLead

    return ecg12aux

# funtion to get the activation time
def get_activation_time(
        Vm : np.ndarray
    ) -> np.ndarray :
    """
    Get the activation time from the Vm.
    
    Parameters
    ----------
    Vm : np.ndarray
        Vm.
        
    Returns
    -------
    actTime : np.ndarray
        Activation time.
    """
    actTime = []
    # check that Vm has 75 columns
    if Vm.shape[1] != 75:
        print('Error: Vm does not have 75 columns')
        return actTime
    for col in range(0,75,1):
        actTime.append(np.argmax(Vm[:,col]>0))
    actTime = np.asarray(actTime)
    actTime = np.reshape(actTime,(75,1))
    return actTime

def fileReader(path: str, finalInd: int, train_test_ratio: float):
    '''
    Args:
        path: Path where the data is residing at the moment
    '''

    # Now, let's load the data itself
    files = []
    regex = r'data_hearts_dd_0p2*'
    pECGTrainData, VmTrainData, pECGTestData, VmTestData, actTimeTrain, actTimeTest  = [], [], [], [], [], []

    for x in os.listdir(path):
        if re.match(regex, x):
            files.append(path + x)

    data_dirs = read_data_dirs(files)[:finalInd]

    trainLength = int(train_test_ratio*len(data_dirs))

    trainIndices = set(np.random.permutation(len(data_dirs))[:trainLength])

    for i, (pECGData_file, VmData_file) in enumerate(tqdm(data_dirs, desc='Loading datafiles ')):
        if i in trainIndices:
            with open(pECGData_file, 'rb') as f:
                pECGTrainData.append(get_standard_leads(np.load(f)))
            with open(VmData_file, 'rb') as f:
                VmTrainData.append(np.load(f))
                actTimeTrain.append(get_activation_time(VmTrainData[-1]).squeeze(1))
        
        else:
            with open(pECGData_file, 'rb') as f:
                pECGTestData.append(get_standard_leads(np.load(f)))
            
            with open(VmData_file, 'rb') as f:
                VmTestData.append(np.load(f))
                actTimeTest.append(get_activation_time(VmTestData[-1]).squeeze(1))
        
    
    VmTrainData = np.stack(VmTrainData, axis = 0)
    pECGTrainData = np.stack(pECGTrainData, axis=0)
    VmTestData = np.stack(VmTestData, axis = 0)
    pECGTestData = np.stack(pECGTestData, axis = 0)
    actTimeTrain = np.stack(actTimeTrain, axis = 0)
    actTimeTest = np.stack(actTimeTest, axis = 0)
    return torch.from_numpy(VmTrainData), torch.from_numpy(pECGTrainData), torch.from_numpy(VmTestData), torch.from_numpy(pECGTestData), torch.from_numpy(actTimeTrain), torch.from_numpy(actTimeTest)


def fileReaderForActivation(path: str, dataInd: int):
    '''
    Load the data and get the activation for the output
    '''
    files = []

    regex = r'data_hearts_dd_0p2*'
    pECGTrainData, ActivationTrainData, pECGTestData, VmTestData  = [], [], [], []
    for x in os.listdir(path):
        if re.match(regex, x):
            files.append(path + x)
    
    data_dirs = read_data_dirs(files)[:dataInd]

    trainLength = int(0.8*len(data_dirs))

    for i, (pECGData_file, VmData_file) in enumerate(tqdm(data_dirs, desc='Loading datafiles ')):
        if i < trainLength:
            with open(pECGData_file, 'rb') as f:
                pECGTrainData.append(get_standard_leads(np.load(f)))
            with open(VmData_file, 'rb') as f:
                ActivationTrainData.append(np.argmax(np.load(f), axis = 0))
        
        else:
            with open(pECGData_file, 'rb') as f:
                pECGTestData.append(get_standard_leads(np.load(f)))
            
            with open(VmData_file, 'rb') as f:
                VmTestData.append(np.load(f))
        
    
    ActivationTrainData = np.stack(ActivationTrainData, axis = 0)
    pECGTrainData = np.stack(pECGTrainData, axis=0)
    VmTestData = np.stack(VmTestData, axis = 0)
    pECGTestData = np.stack(pECGTestData, axis = 0)
    return torch.from_numpy(ActivationTrainData), torch.from_numpy(pECGTrainData), torch.from_numpy(VmTestData), torch.from_numpy(pECGTestData)


