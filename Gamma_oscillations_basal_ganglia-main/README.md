This repository contains the code and simulated data for the paper *"Gamma oscillations in basal ganglia stem from the interplay between local inhibition and beta synchronization"*.  

The repository is divided into three folders:  

- Dopamine depletion:  
  Contains the code for simulating the model with varying dopamine depletion parameters, along with the corresponding analysis and a pickle file containing pre-simulated data.  

- Disconnection:  
  Contains the code for simulating the model by independently disconnecting each projection, the corresponding analysis, and a pickle file containing pre-simulated data.  
  - The pickle file `dd_firing.p` contains the firing rates of each basal ganglia (BG) nucleus, which are required for the disconnection procedure.  

- Isolation:  
  Contains the code for simulating both the original model and the isolated versions of the BG nuclei that display gamma oscillations, the corresponding analysis, and a pickle file containing pre-simulated data.  
  - The pickle file `dd_firing.p` contains the firing rates of each BG nucleus, required for the isolation procedure.  
  - The file `Wavelet_analysis.ipynb` enables a beta-averaged wavelet transform of the activities of the nuclei.  

**Requirements**
The following Python libraries are required to run the code:  
- NumPy  
- SciPy  
- matplotlib  
- pickle  
- neurodsp  
- ANNarchy  
- FOOOF  

