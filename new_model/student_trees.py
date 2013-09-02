import numpy as np
import scipy.stats as stats
import bernoulli_regression_tree as brt

def one_minus(func):
    def new_func(*args,**kwargs):
        return 1-func(*args,**kwargs)
    return new_func

def probit(distances,mu,std,sign,mult):
    results = stats.norm.cdf(distances, mu * (mult ** sign), std)
    if sign > 0:
        return results
    else:
        return 1-results


__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=1,
                                   false_return=0)
on_tree = __contains_node


__distance_node = brt.ContinuousNode(feature='trajector_distance',
                                     function=probit,
                                     parameters=dict(mu=0.15,std=0.05,sign=-1,mult=1))
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__distance_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
near_to_tree = __contains_node

__distance_node = brt.ContinuousNode(feature='trajector_distance',
                                     function=probit,
                                     parameters=dict(mu=0.15,std=0.05,sign=-1,mult=0.75))
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__distance_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
somewhat_near_to_tree = __contains_node

__distance_node = brt.ContinuousNode(feature='trajector_distance',
                                     function=probit,
                                     parameters=dict(mu=0.15,std=0.05,sign=-1,mult=1.5))
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__distance_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
very_near_to_tree = __contains_node


__distance_node = brt.ContinuousNode(feature='trajector_distance',
                                     function=probit,
                                     parameters=dict(mu=0.55,std=0.05,sign=1,mult=1))
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__distance_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
far_from_tree = __contains_node

__distance_node = brt.ContinuousNode(feature='trajector_distance',
                                     function=probit,
                                     parameters=dict(mu=0.55,std=0.05,sign=1,mult=0.75))
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__distance_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
somewhat_far_from_tree = __contains_node

__distance_node = brt.ContinuousNode(feature='trajector_distance',
                                     function=probit,
                                     parameters=dict(mu=0.55,std=0.05,sign=1,mult=1.5))
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__distance_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
very_far_from_tree = __contains_node


__orientation_node = brt.DiscreteNode(decision_feature='trajector_left_of',
                                      decision_operator=np.equal,
                                      decision_value=True,
                                      true_return=1,
                                      false_return=0)
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__orientation_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
to_the_left_of_tree = __contains_node

__orientation_node = brt.DiscreteNode(decision_feature='trajector_right_of',
                                      decision_operator=np.equal,
                                      decision_value=True,
                                      true_return=1,
                                      false_return=0)
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__orientation_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
to_the_right_of_tree = __contains_node

__orientation_node = brt.DiscreteNode(decision_feature='trajector_in_front_of',
                                      decision_operator=np.equal,
                                      decision_value=True,
                                      true_return=1,
                                      false_return=0)
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__orientation_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
in_front_of_tree = __contains_node

__orientation_node = brt.DiscreteNode(decision_feature='trajector_behind',
                                      decision_operator=np.equal,
                                      decision_value=True,
                                      true_return=1,
                                      false_return=0)
__surface_node = brt.DiscreteNode(decision_feature='trajector_surface', 
                                  decision_operator=np.equal, 
                                  decision_value=True,
                                  true_return=0,
                                  false_return=__orientation_node)
__contains_node = brt.DiscreteNode(decision_feature='trajector_contained', 
                                   decision_operator=np.equal, 
                                   decision_value=True,
                                   true_return=0,
                                   false_return=__surface_node)
behind_tree = __contains_node


trees = {
    'on': on_tree,
    'in': on_tree,
    'near to': near_to_tree,
    'somewhat near to': somewhat_near_to_tree,
    'very near to': very_near_to_tree,
    'close to': near_to_tree,
    'somewhat close to': somewhat_near_to_tree,
    'very close to': very_near_to_tree,
    'far from': far_from_tree,
    'somewhat far from': somewhat_far_from_tree,
    'very far from': very_far_from_tree,
    'to the left of': to_the_left_of_tree,
    'to the right of': to_the_right_of_tree,
    'in front of': in_front_of_tree,
    'behind': behind_tree
}