import numpy as np
import pandas as pd
import pickle as pkl
import logging


def prediction(data_dict):
    try:
        data = pd.DataFrame(data=data_dict)
        cols = list(data_dict.columns)

        preprocessor = pkl.load(open('Models\Saved_Models\preprocessor.pkl', 'rb'))
        predictor = pkl.load(open('Models\Saved_Models\model.pkl', 'rb'))

        data_scaled=preprocessor.transform(data)
        data = pd.DataFrame(data_scaled, columns=cols)
        pred = predictor.predict(data)
        logging.info('Success! Prediction done.')

        if pred == 1:
            return 'Bad quality wine.'
        elif pred == 0:
            return 'Good quality wine.'
        
    except:
        logging.error('Error! Prediction failed.')


