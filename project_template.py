import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = ['README.md',
                 'Data/Raw/data_file.csv',
                 'Data/Processed/preprocessed_data.csv',
                 'Data/Features/engineered_features.csv',
                 'Data/Splits/train.csv',
                 'Data/Splits/test.csv',
                 'Data/Splits/Validation.csv',
                 'Notebooks/Exploratory_Analysis.ipynb',
                 'Notebooks/Preprocessing.ipynb',
                 'Notebooks/Feature_Engineering.ipynb',
                 'Notebooks/Model_Training.ipynb',
                 'Notebooks/Inference.ipynb',
                 'Models/Saved_Models/model1.pkl',
                 'Models/Saved_Models/model2.pkl',
                 'Models/Model_Evaluation/evaluation_metrics.txt',
                 'Scripts/Data_Processing.py',
                 'Scripts/Model_Training.py',
                 'Scripts/Inference.py',
                 'Config/Parameters.yaml',
                 'Environment/requirements.txt',
                 'Environment/environment.yaml',
                 'Test/Unit Tests/test_data_processing.py',
                 'Test/Unit Tests/test_model_training.py'
                ]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Created file: {filename}")

    else:
        logging.info(f"File already exists: {filename}")