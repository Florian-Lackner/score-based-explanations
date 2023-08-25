from misc import memoization, timer

from math import factorial
from itertools import combinations, product
from pandas import DataFrame



class Explainer:
    def __init__(self, classifier, preparer, distribution, domains=None):
        self.predict = classifier.predict
        self.preprocess = preparer.preprocess
        self.domains = domains
        self.distribution = distribution



    def partial_entity(self, entity, fixed_features):
        return DataFrame(product(*[[entity[feature]] if feature in fixed_features else self.domains[feature] for feature in entity.index]), columns=entity.index)


    @memoization(key_parameter={"self", "entity"})
    def prediction(self, entity):
        return self.predict(self.preprocess(entity))[0]


    @memoization(key_parameter={"self", "entity", "fixed_features"})
    def expected_prediction(self, entity, fixed_features={}):
        pe = self.partial_entity(entity, fixed_features)
        weight_distribution = pe.apply(lambda e: self.distribution(e, fixed_features), axis=1)
        total_weight = weight_distribution.sum()
        weighted_sum = (self.predict(self.preprocess(pe)) * weight_distribution).sum()

        if total_weight == 0:
            return self.prediction(entity)

        expected_value = weighted_sum / total_weight
        return expected_value


    def counter(self, entity, feature):
        return self.prediction(entity) - self.expected_prediction(entity, set(entity.index)-{feature})


    def counter_plus(self, entity, features):
        return self.prediction(entity) - self.expected_prediction(entity, set(entity.index)-set(features))


    def x_resp(self, entity, feature, max_size=None, get_counter=True):
        entity_prediction = self.prediction(entity)

        candidates = list(set(entity.index)-{feature})
        max_size = len(candidates) if max_size is None else max_size-1
        best_counter = 0
        for size in range(max_size+1):
            for contingency_set in combinations(candidates, size):
                entities = self.partial_entity(entity, set(entity.index)-set(contingency_set))
                bad_entities = entities[self.predict(self.preprocess(entities)) == entity_prediction]
                counters = bad_entities.apply(lambda b_e: self.counter(b_e, feature), axis=1)
                
                best_counter = max(best_counter, max(counters))
            if best_counter > 0:
                return (1/(1+size), best_counter) if get_counter else 1/(1+size)
        return (0.0, 0.0) if get_counter else 0.0


    def resp(self, entity, feature):
        x_resp_score, best_counter = self.x_resp(entity, feature, get_counter=True)
        return x_resp_score * best_counter


    def shap(self, entity, feature):
        return COOP_Game(self.expected_prediction).shapley(entity, {feature})


    def shap_plus(self, entity, features):
        return COOP_Game(self.expected_prediction).shapley(entity, set(features))



class COOP_Game:
    def __init__(self, game_function):
        self.GAME_FUNCTION = game_function

    def shapley(self, all_players, player):
        shapley_value = 0
        for size in range(len(all_players)):
            coefficient = (factorial(size)*factorial(len(all_players)-size-1)) / (factorial(len(all_players)))
            for team in combinations(set(all_players.index)-set(player), size):
                shapley_value += coefficient * (self.GAME_FUNCTION(all_players, set(player).union(team)) - self.GAME_FUNCTION(all_players, team))
        
        return shapley_value