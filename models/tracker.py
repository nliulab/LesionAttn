import numpy as np
from utils.metrics import metric_rules


def is_pareto_improvement(new, best):
    improvement = False
    for metric, rule in metric_rules.items():
        if metric not in new.keys():
            continue
        if rule == 'max':
            if new[metric] > best[metric]:
                improvement = True
            elif new[metric] < best[metric]:
                return False
        elif rule == 'min':
            if new[metric] < best[metric]:
                improvement = True
            elif new[metric] > best[metric]:
                return False
    return improvement


class PerformanceTracker:
    def __init__(self, early_stop_epochs=10, metric="auprc"):
        self.metric = metric
        self.best_metrics = {metric: -np.inf}
        self.best_model_state_dict = None
        self.early_stop_epochs = early_stop_epochs
        self.no_update_epochs = 0
    
    def update(self, metric_dict, model_state_dict):
        if metric_dict[self.metric] > self.best_metrics[self.metric]:
            self.best_metrics = metric_dict
            self.best_model_state_dict = model_state_dict
            self.no_update_epochs = 0
        else:
            self.no_update_epochs += 1

        # check if early stop
        if self.no_update_epochs >= self.early_stop_epochs:
            return False
        else:
            return True

    def export_best_model_state_dict(self):
        return self.best_model_state_dict

    def export_best_metric_dict(self):
        return self.best_metrics


class ParetoFrontierTracker:
    def __init__(self, early_stop_epochs=10, eps_ratio=0.05, metrics=["auprc", "eo"]):
        self.metric_rules = {metric: metric_rules[metric] for metric in metrics}
        self.pareto_front = []
        self.early_stop_epochs = early_stop_epochs
        self.no_update_epochs = 0
        self.eps_ratio = eps_ratio  # make the pareto frontier less strict

    def _is_dominated(self, candidate, pareto_front):
        for other in pareto_front:
            candidate_metrics = candidate["metrics"]
            other_metrics = other["metrics"]
            no_worse = True
            better = False
            for metric, rule in self.metric_rules.items():
                if metric not in other_metrics or metric not in candidate_metrics:
                    continue
                candidate_value = candidate_metrics[metric]
                other_value = other_metrics[metric]
                if rule == 'max':
                    if other_value < candidate_value * (1 - self.eps_ratio):
                        no_worse = False
                    if other_value > candidate_value * (1 + self.eps_ratio):
                        better = True
                elif rule == 'min':
                    if other_value > candidate_value * (1 + self.eps_ratio):
                        no_worse = False
                    if other_value < candidate_value * (1 - self.eps_ratio):
                        better = True
            if no_worse and better:
                return True
        return False

    def _update_pareto_front(self, candidate):
        # Add candidate if it is not dominated
        if not self._is_dominated(candidate, self.pareto_front):
            self.pareto_front.append(candidate)
            # Remove any solutions that are now dominated
            self.pareto_front = [p for p in self.pareto_front if not self._is_dominated(p, [candidate])]
            self.no_update_epochs = 0
        else:
            self.no_update_epochs += 1

    def update(self, metric_dict, model_state_dict):
        candidate = {'metrics': metric_dict, 'model_state_dict': model_state_dict}
        self._update_pareto_front(candidate)

        # check if early stop
        if self.no_update_epochs >= self.early_stop_epochs:
            return False
        else:
            return True

    def export_pareto_front(self):
        return self.pareto_front

    def export_best_model_state_dict(self):
        return self.pareto_front[-1]["model_state_dict"]
    
    def export_best_metric_dict(self):
        return self.pareto_front[-1]["metrics"]
