# Machine Learning Approaches for ECG-Based Heart Condition Analysis and Reconstruction


**If you find this repository useful, we ask that you also acknowledge and cite the original work as detailed in the [Citations](#citations) sections below. Also, don't forget to ⭐ this repository if you find it useful!**

## About This Repository

This repository is a fork of the original [cardiac_ml](https://data-science.llnl.gov/dsc) repository by Mikel Landajuela and collaborators. It has been adapted and modified to better suit specific needs intended for the objectives. 

## Acknowledgments
Special thanks to Mikel Landajuela and his team for their foundational work, which has significantly contributed to the development of this repository. The original codebase and accompanying resources provided an excellent starting point for further exploration and advancement in this field.



## Description
The electrocardiogram (ECG) provides a non-invasive and cost-effective tool for the diagnosis of heart conditions. However, the standard 12-lead ECG is inadequate for mapping out the electrical activity of the heart in sufficient detail for many clinical applications (e.g., identifying the origins of an arrhythmia). In order to construct a more detailed map of the heart, current techniques require not only ECG readings from dozens of locations on a patient’s body, but also patient-specific anatomical models built from expensive medical imaging procedures. For this Data Science Challenge problem, we consider an alternative data-driven approach to reconstructing electroanatomical maps of the heart at clinically relevant resolutions, which combines input from the standard 12-lead electrocardiogram (ECG) with advanced machine learning techniques. We begin with the clearly-defined task of identifying heart conditions from ECG profiles and then consider a range of more open-ended challenges, including the reconstruction of a complete spatio-temporal activation map of the human heart.

<p align="center">
    <img src="figures/rotating_hearts.gif" width=800/>
</p>


## Objective 1 : Heartbeat Classification
We use the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) to access ECG data in order to perform binary classification for healthy heartbeat vs. irregular heartbeat

All relevant notebooks for this objective can be found in the [notebooks/Task1](./notebooks/Task1) folder. 
Begin by exploring the [task_1_getting_started.ipynb](./notebooks/Task1/task_1_getting_started.ipynb) notebook, which guides you through accessing and saving the dataset. Afterward, experiment with various models to tackle this classification task.

## Objective 2 : Irregular Heartbeat Classification
Diagnosing an irregular heartbeat by using the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) to perform multiclass classification to diagnose the irregular heartbeats.

All relevant notebooks for this objective can be found in the [notebooks/Task2](./notebooks/Task2) folder. 
Begin by exploring the [task_2_getting_started.ipynb](./notebooks/Task1/task_2_getting_started.ipynb) notebook, which guides you through accessing and saving the dataset. Afterward, experiment with various models to tackle this classification task.


## Objective 3 : Activation Map Reconstruction from ECG
Sequence-to-vector prediction using the [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106)
to perform activation map reconstruction (i.e. transform a sequence of length 12x500 to 75x1 using a neural network)

All relevant notebooks for this objective can be found in the [notebooks/Task3](./notebooks/Task3) folder. 
Begin by exploring the [task_3_getting_started.ipynb](./notebooks/Task1/task_3_getting_started.ipynb) notebook, which guides you through accessing and saving the dataset. Afterward, experiment with various models to tackle this classification task.


## Objective 4 : Transmembrane Potential Reconstruction from ECG
Sequence-to-sequence prediction using the [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106) to perform transmembrane potential reconstruction (i.e. transform a sequence of length 12x500 to 75x500 using a neural network)

All relevant notebooks for this objective can be found in the [notebooks/Task4](./notebooks/Task4) folder. 
Begin by exploring the [task_4_getting_started.ipynb](./notebooks/Task1/task_4_getting_started.ipynb) notebook, which guides you through accessing and saving the dataset. Afterward, experiment with various models to tackle this classification task.



Additional Information
----------------

### Working with the ECG Heartbeat Categorization Dataset

<details>
<summary>Download dataset</summary>

- Download the dataset from the [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- Unzip the `archive.zip` file
- Rename the folder `archive` as `ecg_dataset` and place it in the root of the git repository

</details>

### Working with the Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals

<details>
<summary>Download dataset</summary>

1. Download the dataset from the [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106)
    - You will need to download all the components of the dataset one by one
2. Unzip the dataset

**Note** : For convenience, we have included a bash script to perform the above steps. To use the script, run the following command from the root of the repository:
```bash 
source download_intracardiac_dataset.sh
```
</details>

<details>




Resources
----------------

- Dataset: [Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals](https://library.ucsd.edu/dc/object/bb29449106)
- Medium Blog post : [Medium Blog post](https://medium.com/p/a20661669937)
- Code Repository : [cardiac_ml](https://github.com/landajuela/cardiac_ml)


Citations
----------------
This repository is supported by the following works:
```
@INPROCEEDINGS{10081783,
  author={Landajuela, Mikel and Anirudh, Rushil and Loscazo, Joe and Blake, Robert},
  booktitle={2022 Computing in Cardiology (CinC)}, 
  title={Intracardiac Electrical Imaging Using the 12-Lead ECG: A Machine Learning Approach Using Synthetic Data}, 
  year={2022},
  volume={498},
  number={},
  pages={1-4},
  keywords={Torso;Measurement;Machine learning algorithms;Imaging;Voltage;Machine learning;Electrocardiography},
  doi={10.22489/CinC.2022.026}}
```
and
```
Landajuela, Mikel; Anirudh, Rushil; Blake, Robert (2022).
Dataset of Simulated Intracardiac Transmembrane Voltage Recordings and ECG Signals. 
In Lawrence Livermore National Laboratory (LLNL) Open Data Initiative. 
UC San Diego Library Digital Collections. 
https://doi.org/10.6075/J0SN094N
```

