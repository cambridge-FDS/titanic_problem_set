# Visalisation Report

## Datasets for visualisation

Train data:
Using train data for for visualisations involving target variable information - e.g. using "Survived" as the hue or target. This is to prevent data leaking in the pre-processing section.

All data:
Using all data for any demographic or feature distribution visualisations that don't involve survival rates.

## Existing Visualisations

Two main visualisation functions were used: count plots and distribution plots. Count plots display the frequency of observations in each category of a variable. Distribution plots show the distribution of continuous numerical variables; it represents the frequency of different value ranges within the feature.

The current fuctions displays two count plots for each feature in [] in both the all data and train data, one with survived as hue and one without. It also displays two distribution plots for each feature in [] from both datasets, one with survived as hue and one without.

## Selected and new visualisations

Using train data for any visualisation incorporating survival rate information, and all data otherwise.
