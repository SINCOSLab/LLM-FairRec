# Can LLMs Enhance Fairness in Recommendation Systems? A Data Augmentation Approach

## The prompt templates

+ load_data_ml1m.py: Loading ml1m data, creating prompts, conducting prompt tuning, and extracting the augmented data. You can adjust the corresponding code (such as evaluation function) freely for your own dataset.
+ generate_response.py: Using the generated prompt to augment interaction data with the API of LLMs.
+ train_gcn_baseline.py: Training the lightgcn model for running fairmi.
+ train_gcn_fairmi.py: Training fairmi and calculating the personal unfairness.

## Requirements
+ scikit-learn 0.24.2
+ scipy 1.5.3
+ openai 0.28.0
+ numpy 1.19.5

## Acknowledgement

Our implementation is based on and inspired by **FairMI**.  
We sincerely thank the authors of FairMI for their open-source contribution.

FairMI GitHub repository: https://github.com/chenzhao-hfut/FairMI

