# Empatica E4, Glucose Data Processing

## Overview
This project is a comprehensive suite for extracting PPG (Photoplethysmography) morphology, RR, and statistical features from Empatica E4 devices. It is structured in a series of steps, each encapsulated in individual Python scripts, to provide a modular and easy-to-understand approach to creating datasets to feed to DL networks. The repository contains code for the TAMU-Sense Project. Contains code from reading the Accelerometer, Blood Volume Pulse, Electrodermal Activity, Heart Rate, Interbeat Interval, and Temperature files from the E4 folder and processing it alongside the glucose file. The code contains all necessary functions, from reading to creating a dataset before feeding it to CNN and RNN networks.

<!-- Here is a more detailed explanation: [slides](https://docs.google.com/presentation/d/1R43jRwrjQzGUOc0ZKuudR8Fe8kVUYRlM1KOL7imoOhY/edit?usp=sharing) -->

## Get Started
### Installation
1. Clone the repository
```bash
git clone https://github.com/david-roh/Sense_glucose.git
cd Sense_glucose
```
2. Install dependent packages
```bash
conda create --name ppg python=3.11
conda activate ppg
pip install -r requirements.txt
```
### Download the Data
We use the data in the **TCH: Cohort 1 Data** and **TCH: Cohort 2 Data** folder from the **[SeNSE TAMU](https://drive.google.com/drive/folders/1Pts4PLTFIYqpPU53k8ZE4H-J4zZNH-WY?usp=drive_link)** team drive.
* Please reach out to Professor [Gutierrez-Osuna](mailto:rgutier@cse.tamu.edu) at the PSI Lab in the Department of Computer Science & Engineering at Texas A&M University if you wish to access the PPG data.

There should be five subject folders (S01-S05) in both **TCH: Cohort 1 Data** and **TCH: Cohort 2 Data** respectively. We only need the ***E4*** and ***cgm*** folders from each.
* For E4: download all folders inside
* For cgm: There should be one Clarity_report_..._.csv file

Download all the data and put them in the same folder. For example:

- SeNSE TAMU
  - Study data
    - TCH: Cohort 1 Data
      - S01
        - e4
          - F1
          - F2
          - A033DE_220628-042642
          - ...
        - cgm
          - Clarity_Export_C01S01_2022-07-05.csv
      - S02
    - TCH: Cohort 2 Data

## Preprocessing
After downloading all the data and placed in the file structure mentioned above, please run the following command for data preprocessing.

This reads all \*.csv files in the folder and combines them in one single pickle file respectively (combined.pkl). It also converts the raw glucose metadata file into the desired glucose file format (glucose.pkl).

### Data Types in Combined Data
The combined data is stored in a DataFrame with the following data types:

- `bvp`, `hr`, `ibi`, `temp`: These columns are stored as `float16` to save memory while providing sufficient precision for the data they represent.
- `acc_x`, `acc_y`, `acc_z`: These columns represent accelerometer data and are stored as `Int8` to save memory.
- `eda`: This column is stored as `float32` to provide sufficient precision for the EDA data.

```bash
python libs/preprocess.py --folder_path <folder_path> --glucose_path <glucose_path>
```
- **folder_path**: The folder path containing all the downloaded E4 files for each subject. 
    - Ex: "./SeNSE TAMU/TCH: Cohort 1 Data/S01/e4".
- **glucose_path**: The path towards the raw glucose metadata file, i.e. the clarity report. 
    - Ex: "./SeNSE TAMU/TCH: Cohort 1 Data/S01/cgm/Clarity_Export_C01S01_2022-07-05.csv"
- out_folder: Optional. If not provided, it is the same as the folder_path. This is where the preprocessed **combined.pkl**, and **glucose.pkl** are saved.

Preview of the combined.csv file:
| datetime                   | bvp  | acc_x | acc_y | acc_z | eda | hr | ibi | temp  |
|----------------------------|:----:|:-----:|:-----:|:-----:|:---:|:--:|:---:|:-----:|
| 2022-06-26 15:57:31.000000 | -0.0 | 38    | 34    | 23    | 0.0 |    |     | 34.66 |
| 2022-06-26 15:57:31.015625 | -0.0 |       |       |       |     |    |     |       |
| 2022-06-26 15:57:31.031250 | -0.0 | 36    | 38    | 30    |     |    |     |       |
| 2022-06-26 15:57:31.046875 | -0.0 |       |       |       |     |    |     |       |

Preview of the glucose.csv file:
| Timestamp               | glucose |
|-------------------------|:-------:|
| 2022-04-02 00:03:05     | 111.0   |
| 2022-04-02 00:09:39     | 105.0   |
| 2022-04-02 00:14:38     | 97.0    |
| 2022-04-02 00:19:39     | 90.0    |



## Process the raw data into PPG beats 
After the preprocessing step is complete, we can begin processing the data by running the command:

```bash
python main.py --input_folder <input_folder> --out_folder <out_folder>
```
- **input_folder**: The folder where the combined.pkl, and glucose.pkl are saved, i.e. the out_folder path in the preprocessing step.
    - Ex: “./SeNSE TAMU/TCH: Cohort 1 Data/S01”.
- **out_folder**: This is the folder where all the final extracted PPG beat data is stored. It will create the cohort and subject folder automatically based on your input_folder.
    - Ex: Create a new folder, **TCH_processed**, under the "SeNSE" folder, i.e. “./SeNSE/TCH_processed”.

The processed ecg beats data (**c1s01.pkl**, **c1s01_hypo.pkl**, **c1s01_normal.pkl**) will be stored under "./SeNSE/TCH_processed/c1s01" for the example above.

Here is an example of the beat extracted.
<p align="center">
    <img src="https://github.com/kathanvyas/Sense_glucose/assets/32810188/7a58b591-615a-4440-90b7-f420e8f40ef1"  width="50%">
</p>

## R peak alignments
We also provide the script for aligning the R peak for all the PPG data.

Run the following command for alignment:
```
python alignment.py --ppg <ppg_file> --all_ppg <all_ppg_folder> --r_peak_pos <target_r_peak_pos> --out_folder <out_folder>
```
- **ppg_file**: The file of the processed PPG beats (from the previous section)
    - Ex: “./SeNSE TAMU/TCH: Cohort 1 Data/S01”.
- **all_ppg_folder**: The folder storing all the processed PPG beats files
    - Ex: "./SeNSE/TCH_processed"
- **target_r_peak_pos**: The position where the aligned R peak should be. If not provided, it will be calculated from the average among all the processed data.
    - Ex: 59 (Mean: 59.41, Std: 14.82 from all the processed ecg data)
- **out_folder**: This is the folder where all the final aligned PPG beat data is stored.
    - Ex: “./SeNSE/TCH_aligned”.

Here are the aligned results:
- Before alignment vs Post Alignment
<p align="center">
    <img src="https://github.com/kathanvyas/Zephyr-BioHArness-Data_Preprocess/assets/32810188/025f1d56-61ef-4c52-9f5f-d213b629cd50"  width="50%">
</p>

- All aligned PPG data
<p align="center">
    <img src="https://github.com/kathanvyas/Zephyr-BioHArness-Data_Preprocess/assets/32810188/660a9e0f-f803-4aa5-a754-91d4c969557f"  width="50%">
</p>

