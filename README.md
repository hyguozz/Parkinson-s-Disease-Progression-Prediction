# Parkinsons-Disease-Progression-Prediction
This is the solution of the problem described in https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/411398

# Introduction
Parkinson’s disease (PD) is a disabling brain disorder that affects movements, cognition, sleep, and other normal functions. Unfortunately, there is no current cure—and the disease worsens over time. It's estimated that by 2037, 1.6 million people in the U.S. will have Parkinson’s disease, at an economic cost approaching $80 billion. Research indicates that protein or peptide abnormalities play a key role in the onset and worsening of this disease. Gaining a better understanding of this—with the help of data science—could provide important clues for the development of new pharmacotherapies to slow the progression or cure Parkinson’s disease.

The goal of this program is to predict MDS-UPDR scores, which measure progression in patients with Parkinson's disease. The Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS) is a comprehensive assessment of both motor and non-motor symptoms associated with Parkinson's. I developed a model trained on data of protein and peptide levels over time in subjects with Parkinson’s disease versus normal age-matched control subjects to predict MDS-UPDRS scores of Parkinsons patients.
# Data and Data preprocessing
## Data description
The goal of this competition is to predict the course of Parkinson's disease (PD) using protein abundance data. The complete set of proteins involved in PD remains an open research question and any proteins that have predictive value are likely worth investigating further. The core of the dataset consists of protein abundance values derived from mass spectrometry readings of cerebrospinal fluid (CSF) samples gathered from several hundred patients. Each patient contributed several samples over the course of multiple years while they also took assessments of PD severity.
1. train_peptides.csv Mass spectrometry data at the peptide level. Peptides are the component subunits of proteins.
2. train_proteins.csv Protein expression frequencies aggregated from the peptide level data.
3. train_clinical_data.csv
4. supplemental_clinical_data.csv Clinical records without any associated CSF samples. This data is intended to provide additional context about the typical progression of Parkinsons. 
## Data preprocessing
1. There are 1038/2615 rows in Train_clinical_df have missing values. In order to make use of all the information given in the train_clinical_df, we will use KNN imputation to fill the missing values.
2. We merge the complete records (no missing values) from Supplementary_clinical_df to Train_clinical_df to augment the clinical dataset aiming to improve the performance of KNN imputation.
3. Apply x = log (x+1) to transform the data into normal distribution like data. This can also improve the KNN imputation performance.
### The trend in the visit_month column of the train_clinical_data
We plot a line graph of the average UPDRS scores (i.e., updrs_1, updrs_2, updrs_3, and updrs_4) against the visit_month, shown below.
![image](https://github.com/hyguozz/Parkinson-s-Disease-Progression-Prediction/assets/36547524/4e6d538c-3274-47bb-bbe6-f77e98bd5f66)

### The analysis of the frequence of visit_month
![image](https://github.com/hyguozz/Parkinson-s-Disease-Progression-Prediction/assets/36547524/80eace80-b348-4a9c-a15e-b66a9467d134)

In the figure above, the interval between visit_month is approximately 6 months until the 60th month, after which the data becomes sparse. This suggests that only a few patient_id have long-term records beyond 60 months. Therefore, we will focus only on the data from the 60 months prior to the current time period. Since the target of the prediction is from the current time point to 6, 6, 12 months, it would be helpful to include data up to 72 months to gather as much information as possible in our training dataset.
### Time series data construction
We build the time series for each patient_id from train_df. As a result, for each patient_id, the time range for the patient's visits is [0,72], and the time interval is 6 months. Forward-fill missing values is used to propagate the most recent non-null value forward in time. 
### Target of the training
For each patient visit where a protein/peptide sample was taken, we estimate both their UPDRS scores for that visit and predict their scores for any potential visits 6, 12, and 24 months later. Therefore, we generate the training target columns to represent UPDRS scores at different time intervals, for example, the updrs_1_plus_6_months, updrs_2_plus_6_months, updrs_3_plus_6_months, and updrs_4_plus_6_months are shifted versions of the updrs_1_plus_0_months, updrs_2_plus_0_months, updrs_3_plus_0_months, and updrs_4_plus_0_months columns, respectively, by one visit interval (6 units). Similar transformations are performed to generate columns for 12-month (updrs_1_plus_12_months, updrs_2_plus_12_months, updrs_3_plus_12_months, updrs_4_plus_12_months) and 24-month (updrs_1_plus_24_months, updrs_2_plus_24_months, updrs_3_plus_24_months, updrs_4_plus_24_months) intervals.

The resulting ts_df_updrs DataFrame contains the constructed time series data with columns representing UPDRS scores at different time intervals for each patient.
### Data preprocessing for 'PeptideAbundance'
1. Since the 'PeptideAbundance' column contains large integer features, we apply a logarithmic transformation (log(x+1)) to approximate normality and stabilize their variance. This transformation is commonly used to address skewness in data with a wide range of values. By applying this transformation, the values in these columns are shifted towards zero, resulting in a more symmetric distribution that is characteristic of normal distribution.

2. To ensure the transformed values have similar scales, we standardize the data using Z-score normalization on the log-transformed values of the original feature. This step ensures that each feature contributes equally to the analysis, regardless of their original scales.

Overall, applying log(x+1) to 'PeptideAbundance' columns and rescaling using Z-score normalization can improve the accuracy and reliability of subsequent analyses.
# Protein-peptide network analysis
Protein-peptide network analysis is a powerful tool for understanding the interactions between proteins and peptides in biological systems. It involves the creation of a network graph, where nodes represent proteins and peptides and edges represent their interactions or relationships. In this project, we can create a network graph where each protein and peptide is represented as a node and their interactions are represented as edges. The weight of the edges are the PeptideAbundance.

A subset of UniProt, which are shared in all visit_id samples are selected. 
1. Among all samples, there are 27 UniProt entries that are present. 
2. Select the 27 UniProt as features and filter the Prot_Peptides_df to have the intersection. 
![image](https://github.com/hyguozz/Parkinson-s-Disease-Progression-Prediction/assets/36547524/8cda2b23-8b42-4300-a23b-4cbbc0e43d7e)
![image](https://github.com/hyguozz/Parkinson-s-Disease-Progression-Prediction/assets/36547524/38eef5b7-453d-4287-a909-73e9dc8131a3)


# Feature extractoin of UniProt-Peptides Network
Principal component analysis (PCA) is used to extract the most important components of the Protein-Peptides network. These components could then be used as features for downstream analysis.

# Deep learning model
The features extracted from the UniPro-Peptide network are inputted into a MLP deep learning model consisting of five layers, each layer containing 64 neurons. 

## Symmetric mean absolute percentage error
Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors. It is usually defined[citation needed] as follows:
![image](https://github.com/hyguozz/Parkinson-s-Disease-Progression-Prediction/assets/36547524/2ae93cfc-0972-4755-bb7d-717206e0740c)
where At is the actual value and Ft is the forecast value.


