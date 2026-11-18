import h5py
import numpy as np
import torch
import pandas as pd

from hparams import HParams


default_class_weights = HParams.__fields__['weights'].default
n_classes=HParams.__fields__['n_classes'].default
outcome_labels = list(default_class_weights.keys())

class H5ECG(torch.utils.data.Dataset):
    def __init__(self, path, target_file,n_classes, mean_=None, std_=None ,transform=None, gsd=str):
        """
        Arguments:
            path (string): Path to the ecg file.
            target_file (string): Outcome file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        
        Load ECG signals and corresponding outcome labels, then apply scaling.
    
        """
        # Read outcomes
        self.outcomes = pd.read_csv(target_file, sep=",")
        
        # Check if 'exclude' column exists and drop rows where it equals 1
        if 'exclusion' in self.outcomes.columns:
            self.outcomes = self.outcomes[self.outcomes['exclusion'] != ""]
        
        # self.outcomes.index is a sequence used to align self.outcomes
        self.outcomes["index_o"]=self.outcomes.index

        # Read ecgs
        self.f = h5py.File(path, "r")
        self.hash = pd.DataFrame(self.f["hash"],columns=['hash'])
        self.hash['hash'] = self.hash['hash'].apply(lambda x: x.decode('utf-8'))
        self.hash["index_e"]=self.hash.index
        self.transform = transform

        # Normalize merge keys to string to avoid dtype mismatch (e.g., int64 vs object).
        self.outcomes['ECSE_BK_hash'] = self.outcomes['ECSE_BK_hash'].astype(str).str.strip()
        self.hash['hash'] = self.hash['hash'].astype(str).str.strip()

        self.outcomes= self.outcomes.merge(self.hash,left_on = 'ECSE_BK_hash', right_on = 'hash', how = 'inner')
        self.outcomes= self.outcomes.sort_values(by=['index_e'])

        # Remove ecgs without gsd
        self.ecgs = (np.asarray((self.f["data"]))[self.outcomes["index_e"],:,:])
        
        # Scaling
        if(mean_ is None):
            self.mean_=(self.ecgs.mean())
            self.std_=(self.ecgs.std())
        else:
            self.mean_=mean_
            self.std_=std_
            
        self.ecgs = torch.tensor((self.ecgs - self.mean_) /self.std_)

        # Concatenate the columns dynamically
        self.outcomes["group"] = self.outcomes[outcome_labels].astype(str).agg("".join, axis=1)
        
        # Create lookup table
        self.hash=self.outcomes[["hash","index_o","index_e","group"]]

        # Store outcomes
        self.outcomes = torch.tensor(np.array(self.outcomes[outcome_labels[0:n_classes]])[:])
        
        
        print(f"H5ECG samples {self.hash.shape[0]}")


    def __len__(self):
        return len(self.ecgs)

    def __getitem__(self, idx):
        """
        Returns a sample containing ECG data and target labels at the given index.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            dict: Sample with 'ecg' and 'target' keys.
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ecg = self.ecgs[idx]
        if self.transform is not None:
            ecg = self.transform(ecg)

        if len(ecg.shape) == 3:
            ecg = ecg.squeeze(0)
        target = self.outcomes[idx]
        sample = {"ecg": ecg, "target": target}

        return sample
    


class H5ECGOnly(H5ECG):
    """
    Subclass of H5ECG that returns only ECG signal data without associated labels.
    """
    
    def __init__(self, path, target_file,n_classes, mean_=None, std_=None, transform=None, gsd=str):
        super().__init__( path=path, target_file=target_file,n_classes=n_classes,mean_=mean_,std_=std_, transform=transform, gsd=gsd)        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ecg = self.ecgs[idx]
        if self.transform is not None:
            ecg = self.transform(ecg)

        if len(ecg.shape) == 3:
            ecg = ecg.squeeze(0)
        return ecg 
