from misc import cache_result, timer
from constants import dataset_configurations, classifier_configurations
from classify import Preparer, split_data, train_model, test_model, print_results
from distributions import get_distribution
from Explainer import Explainer

from os import makedirs
from os.path import dirname, realpath, exists
from itertools import product
from more_itertools import powerset
from pandas import read_csv, DataFrame, Series
from time import perf_counter
import logging



DATA_PREFIX = f"{dirname(realpath(__file__))}/data"
RESULTS_PREFIX = f"{dirname(realpath(__file__))}/results"
if not exists(RESULTS_PREFIX):
    makedirs(RESULTS_PREFIX)
FORCE_CLASSIFICATION = False
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

configs = {
    'blood-transfusion-service-center': {
        'classifiers': ['NB_blood'],
        'distributions': ['uniform', 'fully_factorized', 'experimental'],
        'entities': list(product([2, 4, 11, 16, 74], [2, 4, 7, 50], [500, 1000, 1750, 12500], [14, 23, 35, 57, 98])),
        'features': ['Recency', 'Frequency', 'Monetary', 'Time'],
        'single_feature_functions': ['counter', 'x_resp', 'resp', 'shap'],
        'multi_feature_functions': ['counter_plus', 'shap_plus']
    },
    'german_credit_data': {
        'classifiers': ['NB_german'],
        'distributions': ['uniform', 'fully_factorized', 'experimental'],
        'entities': [
            [22, 'female', 2, 'own', 'little', 'moderate', 5951, 48, 'radio/TV'], [53, 'male', 2, 'free', 'little', 'little', 4870, 24, 'car'], [28, 'male', 3, 'own', 'little', 'moderate', 5234, 30, 'car'], [24, 'female', 2, 'rent', 'little', 'little', 4308, 48, 'business'], [44, 'female', 3, 'free', 'little', 'moderate', 12579, 24, 'car'], [63, 'male', 2, 'own', 'little', 'little', 6836, 60, 'business'], [24, 'male', 2, 'rent', 'moderate', 'little', 6187, 30, 'car'], [58, 'female', 1, 'free', 'little', 'little', 6143, 48, 'car'], [23, 'female', 1, 'rent', 'little', 'little', 6229, 36, 'furniture/equipment'], [25, 'male', 2, 'own', 'little', 'moderate', 14421, 48, 'business'], [47, 'male', 2, 'free', 'moderate', 'moderate', 12612, 36, 'education'], [58, 'male', 2, 'rent', 'little', 'moderate', 15945, 54, 'business'], [23, 'female', 2, 'own', 'quite rich', 'little', 4281, 33, 'furniture/equipment'], [29, 'male', 2, 'own', 'little', 'little', 6887, 36, 'education'], [33, 'male', 2, 'rent', 'little', 'little', 950, 15, 'car'], [30, 'male', 3, 'own', 'little', 'moderate', 4455, 36, 'business'], [50, 'male', 2, 'own', 'little', 'little', 5293, 27, 'business'], [24, 'male', 2, 'free', 'little', 'little', 4605, 48, 'car'], [26, 'male', 2, 'own', 'little', 'little', 4788, 48, 'car'], [28, 'male', 3, 'rent', 'little', 'moderate', 9398, 36, 'car'], [26, 'female', 2, 'own', 'little', 'moderate', 9960, 48, 'furniture/equipment'], [28, 'male', 3, 'own', 'little', 'moderate', 4249, 30, 'car'], [24, 'female', 3, 'own', 'moderate', 'moderate', 7408, 60, 'car'], [23, 'male', 2, 'rent', 'little', 'little', 4110, 24, 'furniture/equipment'], [23, 'female', 2, 'rent', 'little', 'little', 2406, 30, 'furniture/equipment'], [60, 'female', 3, 'free', 'moderate', 'moderate', 14782, 60, 'vacation/others'], [37, 'female', 2, 'rent', 'little', 'little', 7685, 48, 'business'], [57, 'male', 3, 'free', 'little', 'moderate', 14318, 36, 'car'], [23, 'female', 2, 'rent', 'none', 'little', 8471, 18, 'education'], [30, 'female', 3, 'own', 'little', 'moderate', 5096, 48, 'furniture/equipment'], [26, 'female', 2, 'rent', 'little', 'little', 3114, 18, 'furniture/equipment'], [42, 'female', 3, 'free', 'little', 'moderate', 8318, 27, 'business'], [29, 'male', 3, 'rent', 'moderate', 'moderate', 9034, 36, 'furniture/equipment'], [24, 'female', 2, 'rent', 'little', 'little', 1207, 24, 'car'], [53, 'male', 2, 'free', 'little', 'little', 7119, 48, 'furniture/equipment'], [31, 'male', 2, 'rent', 'little', 'little', 2302, 36, 'radio/TV'], [42, 'male', 3, 'free', 'little', 'little', 7763, 48, 'car'], [31, 'female', 2, 'own', 'little', 'little', 6758, 48, 'radio/TV'], [23, 'female', 1, 'rent', 'little', 'little', 3234, 24, 'furniture/equipment'], [20, 'female', 2, 'rent', 'little', 'little', 2039, 18, 'furniture/equipment'], [23, 'female', 2, 'rent', 'little', 'little', 1442, 24, 'car'], [21, 'female', 2, 'rent', 'moderate', 'moderate', 3441, 30, 'furniture/equipment'], [26, 'male', 2, 'own', 'little', 'little', 4370, 42, 'radio/TV'], [32, 'male', 2, 'own', 'little', 'little', 4583, 30, 'furniture/equipment'], [28, 'female', 2, 'own', 'little', 'moderate', 4221, 30, 'business'], [46, 'male', 2, 'free', 'little', 'little', 6331, 48, 'car'], [27, 'male', 3, 'own', 'little', 'moderate', 14027, 60, 'car'], [59, 'female', 2, 'rent', 'little', 'moderate', 6416, 48, 'business'], [26, 'female', 1, 'rent', 'moderate', 'moderate', 4280, 30, 'business'], [24, 'female', 2, 'rent', 'little', 'little', 2124, 18, 'furniture/equipment'], [50, 'male', 2, 'free', 'little', 'moderate', 6224, 48, 'education'], [27, 'male', 2, 'own', 'little', 'little', 5998, 40, 'education'], [24, 'male', 2, 'own', 'little', 'little', 9271, 36, 'car'], [55, 'male', 3, 'free', 'little', 'moderate', 9283, 42, 'car'], [24, 'male', 2, 'own', 'little', 'little', 9629, 36, 'car'], [33, 'female', 2, 'rent', 'little', 'little', 3966, 18, 'car'], [23, 'female', 2, 'rent', 'little', 'little', 1216, 18, 'car'], [29, 'male', 2, 'rent', 'little', 'little', 11816, 45, 'business'], [29, 'male', 2, 'own', 'little', 'little', 5179, 36, 'furniture/equipment'], [24, 'female', 2, 'rent', 'little', 'little', 652, 12, 'furniture/equipment'], [23, 'male', 2, 'own', 'little', 'moderate', 15672, 48, 'business'], [32, 'female', 3, 'own', 'little', 'moderate', 18424, 48, 'vacation/others'], [39, 'male', 2, 'free', 'little', 'little', 10297, 48, 'car'], [42, 'male', 2, 'free', 'little', 'moderate', 6288, 60, 'education'], [29, 'female', 0, 'rent', 'little', 'little', 1193, 24, 'car'], [36, 'male', 2, 'rent', 'little', 'little', 7297, 60, 'business']
        ],
        'features': ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose'],
        'single_feature_functions': ['counter', 'x_resp', 'resp', 'shap'],
        'multi_feature_functions': ['counter_plus', 'shap_plus']
    },
    'creditworthiness': {
        'classifiers': ['random_forest'],
        'distributions': ['uniform', 'fully_factorized'],
        'entities': [
            ['0<=X<200', 48, 'existing paid', 'radio/tv', 5951.0, '<100', '1<=X<4', '2', 'female div/dep/mar', 'none', 2, 'real estate', 22, 'none', 'own', 1, 'skilled', 1, 'none', 'yes'], ['<0', 24, 'delayed previously', 'new car', 4870.0, '<100', '1<=X<4', '3', 'male single', 'none', 4, 'no known property', 53, 'none', 'for free', 2, 'skilled', 2, 'none', 'yes'], ['0<=X<200', 30, 'critical/other existing credit', 'new car', 5234.0, '<100', 'unemployed', 4, 'male mar/wid', 'none', 2, 'car', 28, 'none', 'own', 2, 'high qualif/self emp/mgmt', 1, 'none', 'yes'], ['<0', 48, 'existing paid', 'business', 4308.0, '<100', '<1', '3', 'female div/dep/mar', 'none', 4, 'life insurance', 24, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 24, 'existing paid', 'used car', 12579.0, '<100', '>=7', '4', 'female div/dep/mar', 'none', 2, 'no known property', 44, 'none', 'for free', 1, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['<0', 60, 'delayed previously', 'business', 6836.0, '<100', '>=7', '3', 'male single', 'none', 4, 'no known property', 63, 'none', 'own', 2, 'skilled', 1, 'yes', 'yes'], ['<0', 30, 'critical/other existing credit', 'used car', 6187.0, '100<=X<500', '4<=X<7', '1', 'male mar/wid', 'none', 4, 'car', 24, 'none', 'rent', 2, 'skilled', 1, 'none', 'yes'], ['<0', 48, 'critical/other existing credit', 'used car', 6143.0, '<100', '>=7', '4', 'female div/dep/mar', 'none', 4, 'no known property', 58, 'stores', 'for free', 2, 'unskilled resident', 1, 'none', 'yes'], ['<0', 36, 'critical/other existing credit', 'furniture/equipment', 6229.0, '<100', '<1', '4', 'female div/dep/mar', 'co applicant', 4, 'no known property', 23, 'none', 'rent', 2, 'unskilled resident', 1, 'yes', 'yes'], ['0<=X<200', 48, 'no credits/all paid', 'business', 14421.0, '<100', '1<=X<4', '2', 'male single', 'none', 2, 'car', 25, 'none', 'own', 1, 'skilled', 1, 'yes', 'yes'], ['0<=X<200', 36, 'existing paid', 'education', 12612.0, '100<=X<500', '1<=X<4', '1', 'male single', 'none', 4, 'no known property', 47, 'none', 'for free', 1, 'skilled', 2, 'yes', 'yes'], ['0<=X<200', 54, 'no credits/all paid', 'business', 15945.0, '<100', '<1', '3', 'male single', 'none', 4, 'no known property', 58, 'none', 'rent', 1, 'skilled', 1, 'yes', 'yes'], ['<0', 33, 'critical/other existing credit', 'furniture/equipment', 4281.0, '500<=X<1000', '1<=X<4', '1', 'female div/dep/mar', 'none', 4, 'car', 23, 'none', 'own', 2, 'skilled', 1, 'none', 'yes'], ['<0', 36, 'delayed previously', 'education', 6887.0, '<100', '1<=X<4', '4', 'male single', 'none', 3, 'life insurance', 29, 'stores', 'own', 1, 'skilled', 1, 'yes', 'yes'], ['<0', 15, 'no credits/all paid', 'new car', 950.0, '<100', '>=7', '4', 'male single', 'none', 3, 'car', 33, 'none', 'rent', 2, 'skilled', 2, 'none', 'yes'], ['0<=X<200', 36, 'delayed previously', 'business', 4455.0, '<100', '1<=X<4', '2', 'male div/sep', 'none', 2, 'real estate', 30, 'stores', 'own', 2, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['<0', 27, 'no credits/all paid', 'business', 5293.0, '<100', 'unemployed', 2, 'male single', 'none', 4, 'life insurance', 50, 'stores', 'own', 2, 'skilled', 1, 'yes', 'yes'], ['<0', 48, 'no credits/all paid', 'used car', 4605.0, '<100', '>=7', '3', 'male single', 'none', 4, 'no known property', 24, 'none', 'for free', 2, 'skilled', 2, 'none', 'yes'], ['<0', 48, 'existing paid', 'used car', 4788.0, '<100', '4<=X<7', '4', 'male single', 'none', 3, 'life insurance', 26, 'none', 'own', 1, 'skilled', 2, 'none', 'yes'], ['0<=X<200', 36, 'existing paid', 'used car', 9398.0, '<100', '<1', '1', 'male mar/wid', 'none', 4, 'car', 28, 'none', 'rent', 1, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['0<=X<200', 48, 'existing paid', 'furniture/equipment', 9960.0, '<100', '<1', '1', 'female div/dep/mar', 'none', 2, 'car', 26, 'none', 'own', 1, 'skilled', 1, 'yes', 'yes'], ['0<=X<200', 30, 'critical/other existing credit', 'new car', 4249.0, '<100', 'unemployed', 4, 'male mar/wid', 'none', 2, 'car', 28, 'none', 'own', 2, 'high qualif/self emp/mgmt', 1, 'none', 'yes'], ['0<=X<200', 60, 'existing paid', 'new car', 7408.0, '100<=X<500', '<1', '4', 'female div/dep/mar', 'none', 2, 'life insurance', 24, 'none', 'own', 1, 'high qualif/self emp/mgmt', 1, 'none', 'yes'], ['<0', 24, 'no credits/all paid', 'furniture/equipment', 4110.0, '<100', '>=7', '3', 'male single', 'none', 4, 'no known property', 23, 'bank', 'rent', 2, 'skilled', 2, 'none', 'yes'], ['<0', 30, 'existing paid', 'furniture/equipment', 2406.0, '<100', '4<=X<7', '4', 'female div/dep/mar', 'none', 4, 'real estate', 23, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 60, 'all paid', 'other', 14782.0, '100<=X<500', '>=7', '3', 'female div/dep/mar', 'none', 4, 'no known property', 60, 'bank', 'for free', 2, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['<0', 48, 'all paid', 'business', 7685.0, '<100', '4<=X<7', '2', 'female div/dep/mar', 'guarantor', 4, 'car', 37, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 36, 'existing paid', 'new car', 14318.0, '<100', '>=7', '4', 'male single', 'none', 2, 'no known property', 57, 'none', 'for free', 1, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['<0', 18, 'delayed previously', 'education', 8471.0, 'no known savings', '1<=X<4', '1', 'female div/dep/mar', 'none', 2, 'car', 23, 'none', 'rent', 2, 'skilled', 1, 'yes', 'yes'], ['0<=X<200', 48, 'critical/other existing credit', 'furniture/equipment', 5096.0, '<100', '1<=X<4', '2', 'female div/dep/mar', 'none', 3, 'car', 30, 'none', 'own', 1, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['<0', 18, 'no credits/all paid', 'furniture/equipment', 3114.0, '<100', '<1', '1', 'female div/dep/mar', 'none', 4, 'life insurance', 26, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 27, 'no credits/all paid', 'business', 8318.0, '<100', '>=7', '2', 'female div/dep/mar', 'none', 4, 'no known property', 42, 'none', 'for free', 2, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['0<=X<200', 36, 'existing paid', 'furniture/equipment', 9034.0, '100<=X<500', '<1', '4', 'male single', 'co applicant', 1, 'no known property', 29, 'none', 'rent', 1, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['<0', 24, 'existing paid', 'new car', 1207.0, '<100', '<1', '4', 'female div/dep/mar', 'none', 4, 'life insurance', 24, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['<0', 48, 'no credits/all paid', 'furniture/equipment', 7119.0, '<100', '1<=X<4', '3', 'male single', 'none', 4, 'no known property', 53, 'none', 'for free', 2, 'skilled', 2, 'none', 'yes'], ['<0', 36, 'existing paid', 'radio/tv', 2302.0, '<100', '1<=X<4', '4', 'male div/sep', 'none', 4, 'car', 31, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['<0', 48, 'existing paid', 'new car', 7763.0, '<100', '>=7', '4', 'male single', 'none', 4, 'no known property', 42, 'bank', 'for free', 1, 'high qualif/self emp/mgmt', 1, 'none', 'yes'], ['<0', 48, 'existing paid', 'radio/tv', 6758.0, '<100', '1<=X<4', '3', 'female div/dep/mar', 'none', 2, 'car', 31, 'none', 'own', 1, 'skilled', 1, 'yes', 'yes'], ['<0', 24, 'existing paid', 'furniture/equipment', 3234.0, '<100', '<1', '4', 'female div/dep/mar', 'none', 4, 'real estate', 23, 'none', 'rent', 1, 'unskilled resident', 1, 'yes', 'yes'], ['<0', 18, 'existing paid', 'furniture/equipment', 2039.0, '<100', '1<=X<4', '1', 'female div/dep/mar', 'none', 4, 'real estate', 20, 'bank', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['<0', 24, 'existing paid', 'new car', 1442.0, '<100', '4<=X<7', '4', 'female div/dep/mar', 'none', 4, 'car', 23, 'none', 'rent', 2, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 30, 'existing paid', 'furniture/equipment', 3441.0, '100<=X<500', '1<=X<4', '2', 'female div/dep/mar', 'co applicant', 4, 'car', 21, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['<0', 42, 'delayed previously', 'radio/tv', 4370.0, '<100', '4<=X<7', '3', 'male single', 'none', 2, 'life insurance', 26, 'bank', 'own', 2, 'skilled', 2, 'yes', 'yes'], ['<0', 30, 'no credits/all paid', 'furniture/equipment', 4583.0, '<100', '1<=X<4', '2', 'male div/sep', 'guarantor', 2, 'real estate', 32, 'none', 'own', 2, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 30, 'no credits/all paid', 'business', 4221.0, '<100', '1<=X<4', '2', 'female div/dep/mar', 'none', 1, 'car', 28, 'none', 'own', 2, 'skilled', 1, 'none', 'yes'], ['<0', 48, 'critical/other existing credit', 'used car', 6331.0, '<100', '>=7', '4', 'male single', 'none', 4, 'no known property', 46, 'none', 'for free', 2, 'skilled', 1, 'yes', 'yes'], ['0<=X<200', 60, 'existing paid', 'new car', 14027.0, '<100', '4<=X<7', '4', 'male single', 'none', 2, 'no known property', 27, 'none', 'own', 1, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['0<=X<200', 48, 'all paid', 'business', 6416.0, '<100', '>=7', '4', 'female div/dep/mar', 'none', 3, 'no known property', 59, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 30, 'no credits/all paid', 'business', 4280.0, '100<=X<500', '1<=X<4', '4', 'female div/dep/mar', 'none', 4, 'car', 26, 'none', 'rent', 2, 'unskilled resident', 1, 'none', 'yes'], ['<0', 18, 'critical/other existing credit', 'furniture/equipment', 2124.0, '<100', '1<=X<4', '4', 'female div/dep/mar', 'none', 4, 'real estate', 24, 'none', 'rent', 2, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 48, 'delayed previously', 'education', 6224.0, '<100', '>=7', '4', 'male single', 'none', 4, 'no known property', 50, 'none', 'for free', 1, 'skilled', 1, 'none', 'yes'], ['<0', 40, 'critical/other existing credit', 'education', 5998.0, '<100', '1<=X<4', '4', 'male single', 'none', 3, 'no known property', 27, 'bank', 'own', 1, 'skilled', 1, 'yes', 'yes'], ['<0', 36, 'existing paid', 'new car', 9271.0, '<100', '4<=X<7', '2', 'male single', 'none', 1, 'car', 24, 'none', 'own', 1, 'skilled', 1, 'yes', 'yes'], ['0<=X<200', 42, 'all paid', 'used car', 9283.0, '<100', 'unemployed', 1, 'male single', 'none', 2, 'no known property', 55, 'bank', 'for free', 1, 'high qualif/self emp/mgmt', 1, 'yes', 'yes'], ['<0', 36, 'critical/other existing credit', 'used car', 9629.0, '<100', '4<=X<7', '4', 'male single', 'none', 4, 'car', 24, 'none', 'own', 2, 'skilled', 1, 'yes', 'yes'], ['<0', 18, 'critical/other existing credit', 'new car', 3966.0, '<100', '>=7', '1', 'female div/dep/mar', 'none', 4, 'real estate', 33, 'bank', 'rent', 3, 'skilled', 1, 'yes', 'yes'], ['<0', 18, 'existing paid', 'new car', 1216.0, '<100', '<1', '4', 'female div/dep/mar', 'none', 3, 'car', 23, 'none', 'rent', 1, 'skilled', 1, 'yes', 'yes'], ['<0', 45, 'no credits/all paid', 'business', 11816.0, '<100', '>=7', '2', 'male single', 'none', 4, 'car', 29, 'none', 'rent', 2, 'skilled', 1, 'none', 'yes'], ['<0', 36, 'existing paid', 'furniture/equipment', 5179.0, '<100', '4<=X<7', '4', 'male single', 'none', 2, 'life insurance', 29, 'none', 'own', 1, 'skilled', 1, 'none', 'yes'], ['<0', 12, 'existing paid', 'furniture/equipment', 652.0, '<100', '>=7', '4', 'female div/dep/mar', 'none', 4, 'life insurance', 24, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes'], ['0<=X<200', 48, 'existing paid', 'business', 15672.0, '<100', '1<=X<4', '2', 'male single', 'none', 2, 'car', 23, 'none', 'own', 1, 'skilled', 1, 'yes', 'yes'], ['0<=X<200', 48, 'no credits/all paid', 'other', 18424.0, '<100', '1<=X<4', '1', 'female div/dep/mar', 'none', 2, 'life insurance', 32, 'bank', 'own', 1, 'high qualif/self emp/mgmt', 1, 'yes', 'no'], ['<0', 48, 'existing paid', 'used car', 10297.0, '<100', '4<=X<7', '4', 'male single', 'none', 4, 'no known property', 39, 'stores', 'for free', 3, 'skilled', 2, 'yes', 'yes'], ['0<=X<200', 60, 'existing paid', 'education', 6288.0, '<100', '1<=X<4', '4', 'male single', 'none', 4, 'no known property', 42, 'none', 'for free', 1, 'skilled', 1, 'none', 'yes'], ['<0', 24, 'all paid', 'new car', 1193.0, '<100', 'unemployed', 1, 'female div/dep/mar', 'co applicant', 4, 'no known property', 29, 'none', 'rent', 2, 'unemp/unskilled non res', 1, 'none', 'yes'], ['<0', 60, 'existing paid', 'business', 7297.0, '<100', '>=7', '4', 'male single', 'co applicant', 4, 'no known property', 36, 'none', 'rent', 1, 'skilled', 1, 'none', 'yes']
        ],
        'features': ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status', 'employment', 'installment_commitment', 'personal_status', 'other_parties', 'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone', 'foreign_worker'],
        'single_feature_functions': ['counter', 'x_resp', 'resp'],
        'multi_feature_functions': []
    }
}



@timer
def main(configs):
    for dataset_name in configs.keys():
        dataset_config = dataset_configurations[dataset_name]
        logging.debug(f"DATASET: {dataset_name}")

        data = load_data(dataset_config['load'])
        preparer, data_pp = get_preparer(dataset_config, data)

        if ('preprocessing' in dataset_config) and ('drop' in dataset_config['preprocessing']):
            data = data.drop(columns=dataset_config['preprocessing']['drop'], errors='ignore')
        if ('split' in dataset_config) and ('target' in dataset_config['split']):
            data = data.drop(columns=dataset_config['split']['target'], errors='ignore')
        domains = get_domains(data, dataset_config)

        for classifier_name in configs[dataset_name]['classifiers']:
            classifier_config = classifier_configurations[classifier_name]
            classifier_class = classifier_config['classifier_class'].__name__ if 'classifier_class' in classifier_config else classifier_name
            logging.debug(f"CLASSIFIER: {classifier_class}")
            classifier_filename = f"{DATA_PREFIX}/{dataset_name}.{classifier_class}.model"
            classifier = get_classifier(classifier_config, data_pp, dataset_config, cache=(classifier_filename, "pickle"), force=FORCE_CLASSIFICATION)

            entities = []
            for entity in configs[dataset_name]['entities'].copy():
                logging.debug(f"ENTITY: {entity}\n")
                if 'discretization' in dataset_config['load']:
                    prepared_entity = preparer.discretize_data(DataFrame([entity], columns=data.columns), dataset_config['load']['discretization']).squeeze()
                else:
                    prepared_entity = Series(entity, index=data.columns)
                if classifier.predict(preparer.preprocess(prepared_entity))[0] != 1:
                    logging.warning(f"entity: {entity} is not classified with '1' by {classifier_class}, so it is ignored.")
                    configs[dataset_name]['entities'].remove(entity)
                    continue
                entities.append(prepared_entity)

            for distribution_name in configs[dataset_name]['distributions']:
                logging.debug(f"DISTRIBUTION: {distribution_name}")
                distribution = get_distribution(distribution_name, data)

                for function_name in configs[dataset_name]['single_feature_functions']:
                    explainer = get_explainer(classifier, preparer, distribution, domains)
                    score_function = getattr(explainer, function_name)

                    with open(f"{RESULTS_PREFIX}/single_{dataset_name}_{distribution_name}_{function_name}.dat", 'w') as f:
                        f.write(' '.join(configs[dataset_name]['features'])+" Runtime\n")
                        for entity in entities:
                            start_time = perf_counter()
                            scores = [str(score_function(entity, feature)) for feature in configs[dataset_name]['features']]
                            end_time = perf_counter()

                            f.write(' '.join(scores) + f" {(end_time-start_time):.8f}\n")
                            f.flush()

                for function_name in configs[dataset_name]['multi_feature_functions']:
                    explainer = get_explainer(classifier, preparer, distribution, domains)
                    score_function = getattr(explainer, function_name)

                    with open(f"{RESULTS_PREFIX}/multiple_{dataset_name}_{distribution_name}_{function_name}.dat", 'w') as f:
                        feature_combinations = list(powerset(configs[dataset_name]['features']))
                        f.write(' '.join('_'.join(feature_combination) for feature_combination in feature_combinations)+" Runtime\n")
                        for entity in entities:
                            start_time = perf_counter()
                            scores = [str(score_function(entity, feature_combination)) for feature_combination in feature_combinations]
                            end_time = perf_counter()

                            f.write(' '.join(scores) + f" {(end_time-start_time):.8f}\n")
                            f.flush()


def load_data(loading_config):
    if 'data_file' not in loading_config:
        return DataFrame()

    try:
        data = read_csv(**loading_config['data_file'])
    except FileNotFoundError:
        loading_config['data_file']['filepath_or_buffer'] = f"{DATA_PREFIX}/{loading_config['data_file']['filepath_or_buffer']}"
        data = read_csv(**loading_config['data_file'])
    categorized_data = Preparer.discretize_data(data, loading_config['discretization'])

    return categorized_data


def get_preparer(dataset_config, data):
    if 'preprocessing' in dataset_config:
        preparer = Preparer(**dataset_config['preprocessing'])
        data_pp = preparer.preprocess(data)
        return preparer, data_pp
    return Preparer(), data


@cache_result
def get_classifier(classifier_config, data, dataset_config=None):
    if 'trained_classifier' in classifier_config:
        return classifier_config['trained_classifier']

    training_data, testing_data = split_data(data, **dataset_config['split'])
    classifier = train_model(training_data, **classifier_config)
    if testing_data:
        predictions = test_model(classifier, testing_data)
        print_results(predictions, testing_data)

    return classifier


def get_domains(data_without_target, dataset_config):
    if 'domains' in dataset_config['load']:
        return dataset_config['load']['domains']
    else:
        return {f:set(v) for f,v in data_without_target.to_dict('list').items()}


def get_explainer(classifier, preparer, distribution, domains):
    return Explainer(classifier, preparer, distribution, domains)





if __name__ == "__main__":
    main(configs)