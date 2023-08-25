from numpy import array
from pandas import DataFrame, Series, concat, cut, qcut
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


################################################################################
################################# PREPARE DATA #################################
################################################################################


class Preparer:
    def __init__(self, missing={}, one_hot={}, scale={}, drop=[], map_values={}):
        self.not_yet_fit = True
        
        self.missing = missing
        self.one_hot = one_hot
        self.scale = scale
        self.drop = drop
        self.map_values = map_values

    
    @staticmethod
    def discretize_data(data, discretisation={}):
        for feature, (bins,labels) in discretisation.items():
            if type(bins) == int:
                data[feature], bins = qcut(data[feature], bins, retbins=True, labels=labels)
                discretisation[feature] = (bins, labels)
            else:
                data[feature] = cut(data[feature], bins, labels=labels, include_lowest=True)
        
        return data


    @staticmethod
    def transform_data(data, preparation_config, fit, results_in_more_columns=False):
        for encoder,columns in preparation_config.items():
            encoded_data = encoder.fit_transform(data[columns]) if fit else encoder.transform(data[columns])
            if results_in_more_columns:
                one_hot_data = DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(), dtype=bool)
                data = concat([data.drop(columns=columns), one_hot_data], axis=1)
            else:
                data[columns] = encoded_data
        return data


    def preprocess(self, data_pp):
        if type(data_pp) == Series:
            data_pp = data_pp.to_frame().transpose().reset_index(drop=True)

        data_pp = self.transform_data(data_pp, self.missing, self.not_yet_fit)
        data_pp = self.transform_data(data_pp, self.one_hot, self.not_yet_fit, results_in_more_columns=True)
        data_pp = self.transform_data(data_pp, self.scale, self.not_yet_fit)

        if self.drop:
            data_pp = data_pp.drop(columns=self.drop, errors='ignore')
        if self.map_values:
            for feature,mapping in self.map_values.items():
                if feature in data_pp.columns:
                    data_pp[feature] = data_pp[feature].map(mapping)

        self.not_yet_fit = False

        return data_pp



################################################################################
################################## SPLITTING ###################################
################################################################################

def split_data(data, target, test_ratio=0.2, random_state=None):
    if test_ratio == 0:
        data = {'data': data.drop(columns=target), 'target':data[target]}
        return data, None
    else:
        training, testing = train_test_split(data, test_size=test_ratio, random_state=random_state)
        training_data = {'data': training.drop(columns=target), 'target':training[target]}
        testing_data = {'data': testing.drop(columns=target), 'target':testing[target]}
        return training_data, testing_data



################################################################################
################################### TRAINING ###################################
################################################################################

def train_model(training_data, classifier_class, parameter_ranges={}, parameters={}, cv=2):
    # find optimal parameters with grid search
    if parameter_ranges:
        if parameters:
            parameter_ranges.update({parameter: [value] for parameter, value in parameters.items()})
        grid = GridSearchCV(
                estimator=classifier_class(),
                param_grid=parameter_ranges,
                scoring = 'accuracy',
                n_jobs=-1,
                cv = cv,
                verbose = 0
            )
        grid.fit(training_data['data'], training_data['target'])

        print(f"Best Parameter: {grid.best_params_}")
        print(f"Best accuracy from GridSearchCV: {grid.best_score_}")

        plot_parameter_ranges(grid, dict(filter(lambda e: len(e[1])>1, parameter_ranges.items())))

        classifier = grid.best_estimator_

    # train classifier with fixed parameters
    else:
        classifier = classifier_class(**parameters)
        classifier.fit(training_data['data'], training_data['target'])

    return classifier


def plot_parameter_ranges(grid, parameter_ranges):
    if parameter_ranges:
        import matplotlib.pyplot as plt
        from itertools import product

        fig = plt.figure(figsize=(16,9))
        only_interesting = {parameter:values for parameter, values in parameter_ranges.items() if len(values) > 1}

        for i,(parameter, values) in enumerate(only_interesting.items()):
            fig.add_subplot(1,len(only_interesting),i+1)

            annotations = {}
            results = zip(*[grid.cv_results_['mean_test_score'][grid.cv_results_[f'param_{parameter}']==value] for value in values])
            configurations = array(grid.cv_results_['params'])[grid.cv_results_[f'param_{parameter}']==values[0]]
            for configuration, result in zip(configurations, results):
                plt.plot(values, result, marker="o", label=str([val for par,val in sorted(configuration.items()) if par != parameter]))
                annotation = [val for par,val in sorted(configuration.items()) if par != parameter and par in only_interesting]
                position = (values[0], result[0])
                if annotation:
                    annotations[position] = f"{annotations[position]},{annotation}" if position in annotations else annotation
            plt.xlabel(parameter)
            plt.ylabel("mean accuracy")

            for position, annotation in annotations.items():
                plt.annotate(annotation, xy=position)
        plt.show()



################################################################################
################################### TESTING ####################################
################################################################################

def test_model(classifier, testing_data):
    return classifier.predict(testing_data['data'])



################################################################################
#################################### OUTPUT ####################################
################################################################################


def print_results(predictions, testing_data):
    print("Confusion Matrix:")
    cm = confusion_matrix(testing_data['target'], predictions)
    length_of_longest_number = len(str(cm.max()))
    print(f" TP: {str(cm[1][1]).rjust(length_of_longest_number)}, FP: {str(cm[0][1]).rjust(length_of_longest_number)}")
    print(f" FN: {str(cm[1][0]).rjust(length_of_longest_number)}, TN: {str(cm[0][0]).rjust(length_of_longest_number)}")
    print(f"accuracy: {accuracy_score(testing_data['target'], predictions)}")
    print(f"precision: {precision_score(testing_data['target'], predictions)}")
    print(f"recall: {recall_score(testing_data['target'], predictions)}")