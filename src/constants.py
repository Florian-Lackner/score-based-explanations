from distributions import *

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


class SampleClassifier:
    def predict(self, entities):
        return np.array(entities.apply(lambda entity: int((entity['A']!=0 and entity['B']==3) or (entity['A']==0 and entity['B']==1)), axis=1))


dataset_configurations = {
    'blood-transfusion-service-center': { #https://www.openml.org/search?type=data&status=active&sort=runs&qualities.NumberOfFeatures=lte_10&id=1464
        'load': {
            'data_file': {
                'filepath_or_buffer': 'blood-transfusion-service-center.csv'
            },
            'discretization': {
                'Recency': ([0, 2.0, 4.0, 11.0, 16.0, 74.0], ['<3', '3-4', '5-11', '12-16', '>16']),
                'Frequency':([0, 2.0, 4.0, 7.0, 50.0], ['<3', '3-4', '5-7', '>7']),
                'Monetary': ([0, 500.0, 1000.0, 1750.0, 12500.0], ['<501', '501-1000', '1001-1750', '>1750']),
                'Time': ([0, 14.0, 23.0, 35.0, 57.0, 98.0], ['<15', '15-23', '24-35', '36-57', '>57'])
            },
        },
        'preprocessing': {
            'missing': {},
            'one_hot': {
                OneHotEncoder(): [
                    'Recency',
                    'Frequency',
                    'Monetary',
                    'Time'
                ]
            },
            'scale': {},
            'drop': [],
            'map_values': {
                'Donated': {0: 1, 1: 0}
            },
        },
        'split': {
            'target': 'Donated',
            'test_ratio': 0.2,
            'random_state': 123
        }
    },
    'german_credit_data': { #https://www.kaggle.com/datasets/uciml/german-credit
        'load': {
            'data_file': {
                'filepath_or_buffer': 'german_credit_data.csv'
            },
            'discretization': {
                'Age': ([0, 27.0, 33.0, 42.0, 75.0], ['<28', '28-33', '34-42', '>42']),
                #'Job': ([-1,0,1,2,3], ['unskilled and non-resident', 'unskilled and resident', 'skilled', 'highly skilled']),
                'Credit amount': ([0, 1366.0, 2320.0, 3973.0, 20000.0], ['<1367', '1367-2320', '2321-3973', '>3973']),
                'Duration': ([0, 12.0, 24.0, 72.0], ['<13', '13-24', '>24']),
            },
        },
        'preprocessing': {
            'missing': {},
            'one_hot': {
                OneHotEncoder(): [
                    'Age',
                    'Sex',
                    #'Job',
                    'Housing',
                    'Saving accounts',
                    'Checking account',
                    'Credit amount',
                    'Duration',
                    'Purpose'
                ]
            },
            'scale': {},
            'drop': ['ID'],
            'map_values': {
                'Risk': {'good': 0, 'bad': 1}
            },
        },
        'split': {
            'target': 'Risk',
            'test_ratio': 0.2,
            'random_state': 123
        }
    },
    'creditworthiness': { #https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
        'load': {
            'data_file': {
                'filepath_or_buffer': 'creditworthiness.csv',
                'quotechar': "'"
            },
            'discretization': {
                'age': ([0, 27.0, 33.0, 42.0, 75.0], ['<28', '28-33', '34-42', '>42']),
                'credit_amount': ([0, 1366.0, 2320.0, 3973.0, 20000.0], ['<1367', '1367-2320', '2321-3973', '>3973']),
                'duration': ([0, 12.0, 24.0, 72.0], ['<13', '13-24', '>24']),
            },
        },
        'preprocessing': {
            'missing': {},
            'one_hot': {
                OneHotEncoder(): [
                    'checking_status',
                    'credit_history',
                    'purpose',
                    'savings_status',
                    'employment',
                    'personal_status',
                    'other_parties',
                    'property_magnitude',
                    'other_payment_plans',
                    'housing',
                    'job',
                    'own_telephone',
                    'foreign_worker',
                    'duration',
                    'credit_amount',
                    'age'
                ]
            },
            'scale': {},
            'drop': [],
            'map_values': {
                'class': {'good': 0, 'bad': 1}
            },
        },
        'split': {
            'target': 'class',
            'test_ratio': 0.2,
            'random_state': 123
        }
    },
    'resp': {
        'load': {
            'domains': {'A': [0,1,2,3], 'B': [0,1,2,3]}
        },
        'preprocessing': {}
    },
    'fake': {
        'load': {
            'domains': {'A': [0,1,2,3], 'B': [0,1,2,3], 'C': [0,1,2,3]}
        },
        'preprocessing': {}
    }
}


classifier_configurations = {
    'NB_blood': {
        'classifier_class': GaussianNB,
        'parameters': {
            'var_smoothing': 1.484,
        },
        'cv': 10,
    },
    'NB_german': {
        'classifier_class': GaussianNB,
        'parameters': {
            'var_smoothing': 1,
        },
        'cv': 10,
    },
    'random_forest': {
        'classifier_class': RandomForestClassifier,
        'parameters': {
            'n_estimators': 50,
            'criterion' : 'entropy',
            'min_samples_split' :  0.012,
            'random_state': 123
        },
        'cv': 10,
    },
    'fake_classifier': {
        'trained_classifier': SampleClassifier()
    },
    'NB_range': {
        'classifier_class': GaussianNB,
        'parameter_ranges': {
            'var_smoothing': np.linspace(0.1, 10, 100)#np.logspace(1,-10, 100)#np.linspace(2e-8, 3e-8, 100)
        },
        'parameters': {
            #'var_smoothing': 2.25e-8,
        },
        'cv': 10,
    },
    'RF_range': {
        'classifier_class': RandomForestClassifier,
        'parameter_ranges': {
            'n_estimators': np.linspace(10, 100, 10).astype(int).tolist(),
            'criterion' :  ['gini','entropy'],
            'min_samples_split' :  np.linspace(0.012, 0.014, 10),
            'random_state': [123]
        },
        'cv': 10,
    },
    'NB_range_german': {
        'classifier_class': GaussianNB,
        'parameter_ranges': {
            'var_smoothing': np.linspace(0.9, 1.4, 100)#np.logspace(1,-10, 100)#np.linspace(2e-8, 3e-8, 100)
        },
        'parameters': {
            #'var_smoothing': 2.25e-8,
        },
        'cv': 10,
    },
    'RF_range_german': {
        'classifier_class': RandomForestClassifier,
        'parameter_ranges': {
            'n_estimators': np.linspace(10, 100, 10).astype(int).tolist(),
            'criterion' :  ['gini','entropy'],
            'min_samples_split' :  np.linspace(0.09, 0.11, 10),
            'random_state': [123]
        },
        'cv': 10,
    }
}