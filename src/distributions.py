from numpy import prod



def get_distribution(distribution, data):
    if distribution == 'fully_factorized':
        return define_fully_factorized_distribution(data)
    elif distribution == 'experimental':
        return define_experimental_distribution(data)
    elif distribution == 'uniform':
        return uniform_distribution
    else:
        raise Exception('Distribution not defined.')


def define_fully_factorized_distribution(data):
    feature_occurrences = {}
    for domain in data.columns:
        feature_occurrences[domain] = data[domain].groupby(data[domain]).count().to_dict()
    def fully_factorized_distribution(entity, fixed_features):
        return prod([feature_occurrences[feature][entity[feature]] / len(data) for feature in data.columns if feature not in fixed_features])
    return fully_factorized_distribution


def define_experimental_distribution(data):
    occurrences = data.value_counts()

    def experimental_distribution(entity, fixed_features):
        if tuple(entity) in occurrences:
            return occurrences[tuple(entity)]
        return 0
    return experimental_distribution


def uniform_distribution(entity, fixed_features):
    return 1