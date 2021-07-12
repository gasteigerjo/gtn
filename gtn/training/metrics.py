import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score


class Metrics:
    def __init__(
        self, metrics_list, metric_to_stop_on=None, minimize_stop_on=True, patience=10
    ):
        if "loss" not in metrics_list:
            metrics_list.insert(0, "loss")
        self.metrics_list = metrics_list
        self.stop_on = metric_to_stop_on
        if metric_to_stop_on:
            assert metric_to_stop_on in metrics_list
            if minimize_stop_on:
                self.compare = lambda new, best: new < best
                self.best_value = np.inf
            else:
                self.compare = lambda new, best: new > best
                self.best_value = -np.inf
            self.patience = patience
            self.no_improvement = 0
        self.reset()

    def reset(self):
        self.output_list = []
        self.labels_list = []
        self.total_loss = 0

    def set_values(self, loss, hits_1):
        self.total_loss = loss
        self.output_list = [np.array([hits_1])]
        self.labels_list = [np.array([1.0])]

    def update(self, output, labels, mean_loss):
        self.output_list.append(output.detach().cpu().numpy())
        self.labels_list.append(labels.detach().cpu().numpy())
        self.total_loss += mean_loss.item() * labels.size(0)

    def readout(self, metrics_list=None):
        out_metrics = metrics_list or self.metrics_list
        outlist = []
        output = np.concatenate(self.output_list)
        labels = np.concatenate(self.labels_list)
        batch_size = output.size
        for metric in out_metrics:
            if metric == "loss":
                tmp = self.total_loss / batch_size
            elif metric == "rmse":
                tmp = mean_squared_error(labels, output, squared=False)
            elif metric == "cvrmse":
                tmp = mean_squared_error(labels, output, squared=False) / (
                    labels.sum() / batch_size
                )
            elif metric == "label_std":
                tmp = labels.std()
            elif metric == "auc":
                # output -> distance, -output -> similarity
                tmp = roc_auc_score(labels, -output)
            elif metric == "acc":
                # Here output=d(G1, G2) and lables=d(G1, G3)
                # smaller -> more similar
                tmp = (output < labels).sum() / batch_size
            elif metric == "hits@1":
                # output -> number of total hits
                # labels -> number of possible hits
                assert np.all(output <= labels)
                tmp = output.sum() / labels.sum()
            else:
                raise ValueError("not implemented")
            outlist.append(tmp)
        self.epoch_outputs = output
        self.epoch_labels = labels
        return outlist

    def get_log_dict(self, metrics_list=None):
        out_metrics = metrics_list or self.metrics_list
        outlist = self.readout(out_metrics)
        stat = {}
        for idx in range(len(out_metrics)):
            stat[out_metrics[idx]] = outlist[idx]
        return stat

    def get_string(self, metrics_list=None):
        out_metrics = metrics_list or self.metrics_list
        outlist = self.readout(out_metrics)
        outstring = ""
        for idx in range(len(out_metrics)):
            if idx != 0:
                outstring += ", "
            outstring += f"{out_metrics[idx]}: {outlist[idx]:6.4f}"
        return outstring

    def is_best_val(self):
        val = self.readout([self.stop_on])[0]
        if self.compare(val, self.best_value):
            self.best_value = val
            self.no_improvement = 0
            return True
        else:
            self.no_improvement += 1
            return False

    # Call after calling is_best_val
    def is_patience_over(self):
        if self.stop_on:
            return self.patience < self.no_improvement
        else:
            return False

    def get_best_values_string(self):
        return f"{self.stop_on}: {self.best_value}"

    def get_result(self):
        return float(
            self.best_value
        )  # cast to float simply due to appearance in database
