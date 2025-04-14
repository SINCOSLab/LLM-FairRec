# Can LLMs Enhance Fairness in Recommendation Systems? A Data Augmentation Approach

## The prompt templates

+ load_data_ml1m.py / load_data_lastfm.py: Loading data for ml1m / LastFM, creating prompts, conducting prompt tuning, and extracting the augmented data. You can adjust the corresponding code (such as evaluation function) freely for your own dataset.
+ generate_response.py: Using the generated prompt to augment interaction data with the API of LLMs.

## Requirements

+ scikit-learn 0.24.2
+ scipy 1.5.3
+ openai 0.28.0
+ numpy 1.19.5

## Datasets
The raw data of ml1m and LastFM can be downloaded from: https://grouplens.org/datasets/movielens/ and http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html, respectively.



