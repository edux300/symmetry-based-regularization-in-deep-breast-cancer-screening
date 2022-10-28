import copy
import numpy as np
import scipy.special

from engine.main import Callback
from engine.auxiliary import to_numpy

import metrics.classification
from metrics.detection import compute_metrics as compute_detection_metrics
from metrics.detection import compute_recall_at_FPI


class GeneralMetricCallback(Callback):
    def __init__(self, phase, key, compute_fn, scores_key, labels_key):
        assert phase in ["both", "train", "valid"]
        self.reset()
        self.key = key
        self.phase = phase
        self.compute_fn = compute_fn
        self.key = scores_key
        self.labels_key = labels_key

    def process(self):
        self.probs = np.concatenate(self.probs)
        self.labels = np.concatenate(self.labels)

    def reset(self):
        self.probs = []
        self.labels = []

    def compute(self):
        return self.compute_fn(self.labels, self.probs)

    def on_batch_ended(self, resources):
        if self.active:
            self.probs.append(resources[self.scores_key])
            self.labels.append(resources[self.labels_key])

    def on_train_started(self, resources):
        if self.phase in ["both", "train"]:
            self.reset()
            self.active = True
        else:
            self.active = False

    def on_train_ended(self, resources):
        if self.phase in ["both", "train"]:
            self.process()
            resources["average_train_{}".format(self.key)] = self.compute()

    def on_valid_started(self, resources):
        if self.phase in ["both", "valid"]:
            self.reset()
            self.active = True
        else:
            self.active = False

    def on_valid_ended(self, resources):
        if self.phase in ["both", "valid"]:
            self.process()
            resources["average_valid_{}".format(self.key)] = self.compute()

import math
class metricCallback(Callback):
    def __init__(self, phase, key, softmax_first=True, maximize=True):
        assert phase in ["both", "train", "valid"]
        self.reset()
        self.key = key
        self.phase = phase
        self.softmax_first = softmax_first
        self.maximize = maximize
        if maximize:
            self.train_max_so_far = -math.inf
            self.valid_max_so_far = -math.inf
        else:
            self.train_min_so_far = math.inf
            self.valid_min_so_far = math.inf
        self.train_patience = 0
        self.valid_patience = 0

    def process(self):
        self.probs = np.concatenate(self.probs)
        if self.softmax_first:
            self.probs = scipy.special.softmax(self.probs, 1)
        self.labels = np.concatenate(self.labels)

    def reset(self):
        self.probs = []
        self.labels = []

    def compute(self):
        metric = self.compute_fn(self.probs, self.labels)
        if self.curr_phase == "train":
            self.train_patience += 1
            if self.maximize:
                if self.train_max_so_far < metric:
                    self.train_max_so_far = metric
                    self.train_patience = 0
            else:
                if self.train_min_so_far > metric:
                    self.train_min_so_far = metric
                    self.train_patience = 0
        else:
            self.valid_patience += 1
            if self.maximize:
                if self.valid_max_so_far < metric:
                    self.valid_max_so_far = metric
                    self.valid_patience = 0
            else:
                if self.valid_min_so_far > metric:
                    self.valid_min_so_far = metric
                    self.valid_patience = 0
        return metric

    def on_batch_ended(self, resources):
        if self.active:
            self.probs.append(to_numpy(resources["out"]))
            self.labels.append(to_numpy(resources["labels"]))

    def on_train_started(self, resources):
        if self.phase in ["both", "train"]:
            self.reset()
            self.active = True
            self.curr_phase = "train"
        else:
            self.active = False

    def on_train_ended(self, resources):
        if self.phase in ["both", "train"]:
            self.process()
            resources["average_train_{}".format(self.key)] = self.compute()

    def on_valid_started(self, resources):
        if self.phase in ["both", "valid"]:
            self.reset()
            self.active = True
            self.curr_phase = "valid"
        else:
            self.active = False

    def on_valid_ended(self, resources):
        if self.phase in ["both", "valid"]:
            self.process()
            resources["average_valid_{}".format(self.key)] = self.compute()

class batchMetricCallback(Callback):
    def __init__(self, phase, key, softmax_first=True):
        assert phase in ["both", "train", "valid"]
        self.key = key
        self.phase = phase
        self.softmax_first = softmax_first
        self.values = []
        self.weights = []

    def on_batch_ended(self, resources):
        if self.active:
            self.probs = to_numpy(resources["out"])
            self.labels = to_numpy(resources["labels"])
            if self.softmax_first:
                self.probs = scipy.special.softmax(self.probs, 1)
            self.values.append(self.compute_fn(self.probs, self.labels))
            self.weights.append(self.probs.shape[0])

    def on_train_started(self, resources):
        self.active = True if self.phase in ["both", "train"] else False

    def on_valid_started(self, resources):
        self.active = True if self.phase in ["both", "valid"] else False

    def on_train_ended(self, resources):
        if self.phase in ["both", "train"]:
            resources["average_train_{}".format(self.key)] = np.sum((np.array(self.values) * np.array(self.weights))) / np.sum(self.weights)
            self.values = []
            self.weights = []

    def on_valid_ended(self, resources):
        if self.phase in ["both", "valid"]:
            resources["average_valid_{}".format(self.key)] = np.sum((np.array(self.values) * np.array(self.weights))) / np.sum(self.weights)
            self.values = []
            self.weights = []


class BacthAccuracyCallback(batchMetricCallback):
    def __init__(self, phase):
        super(BacthAccuracyCallback, self).__init__(phase, "accuracy")
        self.compute_fn = metrics.classification.accuracy

class BacthTop5AccuracyCallback(batchMetricCallback):
    def __init__(self, phase):
        super(BacthTop5AccuracyCallback, self).__init__(phase, "top5-accuracy")
        self.compute_fn = metrics.classification.top5accuracy

class AccuracyCallback(metricCallback):
    def __init__(self, phase, **kwargs):
        super(AccuracyCallback, self).__init__(phase, "accuracy", **kwargs)
        self.compute_fn = metrics.classification.accuracy

class Top5AccuracyCallback(metricCallback):
    def __init__(self, phase, **kwargs):
        super(Top5AccuracyCallback, self).__init__(phase, "top5-accuracy", **kwargs)
        self.compute_fn = metrics.classification.top5accuracy

class BalancedAccuracyCallback(metricCallback):
    def __init__(self, phase, **kwargs):
        super(BalancedAccuracyCallback, self).__init__(phase, "bal-accuracy", **kwargs)
        self.compute_fn = metrics.classification.bal_accuracy

class RocAUCCallback(metricCallback):
    def __init__(self, phase, negative_class=None, weighted=False, **kwargs):
        super(RocAUCCallback, self).__init__(phase, "rocAUC", **kwargs)
        self.compute_fn = lambda x,y :metrics.classification.rocauc(x,y,
                                                           negative_class=negative_class,
                                                           weighted=weighted)

class F1ScoreCallback(metricCallback):
    def __init__(self, phase, negative_class=None, weighted=False, **kwargs):
        super(F1ScoreCallback, self).__init__(phase, "f1-score", **kwargs)
        self.compute_fn = lambda x,y :metrics.classification.f1score(x,y,
                                                            negative_class=negative_class,
                                                            weighted=weighted)

class AverageCallback(Callback):
    def __init__(self, key, weight_key=None, phase="both"):
        self.key = key
        self.weight_key = weight_key
        self.reset()
        self.active = True
        assert phase in ["both", "train", "valid"]
        self.phase=phase

    def reset(self):
        self.values = []
        self.weights = []

    def compute_mean(self, epsilon=1e-6):
        total_value = 0
        total_weight = epsilon
        for v, w in zip(self.values, self.weights):
            total_value += v*w
            total_weight += w
        return total_value / total_weight

    def on_batch_ended(self, resources):
        if self.active:
            value = to_numpy(resources[self.key])
            weight = resources[self.weight_key] if self.weight_key else 1
            self.values.append(value)
            self.weights.append(weight)

    def on_train_started(self, resources):
        if self.phase in ["both", "train"]:
            self.reset()
            self.active = True
        else:
            self.active = False

    def on_train_ended(self, resources):
        if self.phase in ["both", "train"]:
            resources["average_train_{}".format(self.key)] = self.compute_mean()

    def on_valid_started(self, resources):
        if self.phase in ["both", "valid"]:
            self.reset()
            self.active = True
        else:
            self.active = False

    def on_valid_ended(self, resources):
        if self.phase in ["both", "valid"]:
            resources["average_valid_{}".format(self.key)] = self.compute_mean()

class DetectionRocAUCCallback(Callback):
    def __init__(self, included_classes):
        self.included_classes = included_classes
        self.reset()
        self.active = False

    def reset(self):
        self.scores = []
        self.labels = []

    def on_valid_started(self, resources):
        self.active = True
        self.reset()

    """
    def select(obj, labels, k):
        if k == "all":
            return obj
        if obj.shape[0] > 0:
            return obj[labels == int(k)]
        return obj
    """

    def on_batch_ended(self, resources):
        if self.active:
            _, targets = resources["batch"]
            outs = resources["out"]
            assert len(targets) == len(outs)
            for target, out in zip(targets, outs):
                if len(target["labels"]):
                    label = 1 if np.any([k in target["labels"] for k in self.included_classes]) else 0
                else:
                    label = 0

                score = 0
                if len(out["scores"]):
                    score = np.max(to_numpy(out["scores"]))

                self.scores.append(score)
                self.labels.append(label)


        """
        if not self.active:
            return
        _, targets = resources["batch"]
        outs = resources["out"]

        if len(targets) < len(outs):
            targets = [t for l in targets for t in l]

        assert len(targets) == len(outs)
        for target, out in zip(targets, outs):
            max_score = 0
            global_label = 0
            max_score = 0
            if isinstance(target, list):
                target = target[0]
            for k in self.malignant_classes:
                ground_truth = DetectionRocAUCCallback.select(target["boxes"], target["labels"], k)
                if len(ground_truth) > 0:
                    global_label = 1
                    break
            for k in self.malignant_classes:
                scores = DetectionRocAUCCallback.select(out["scores"], out["labels"], k)
                if len(scores) != 0:
                    max_score = max(max(scores), max_score)

            self.scores.append(max_score)
            self.labels.append(global_label)

        """

    def on_valid_ended(self, resources):
        self.active = False
        resources["epoch_valid_rocAUC"] = self.compute_rocauc()
        resources["epoch_valid_accuracy"] = self.compute_accuracy()

    def compute_rocauc(self):
        return metrics.classification.rocauc(np.array(self.scores), np.array(self.labels))

    def compute_accuracy(self):
        return ((np.array(self.scores)<0.5) == (np.array(self.labels)<0.5)).mean()



def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

"""
Input Arguments:
    num_classes      : number of classes considered (including background if reserve_0)
    reserve_0        : true if class 0 is "background"
    keys_and_classes : a mapping {str: list}. Each entry will be used to compute
                       the Recall@FPI, considering the classes in list positive
                       and the remaining negative
    iou              : intersection over union threshold for correct detection

todo:
    ious as a list
    object size information maybe helpful in debugging models

"""
class DetectionValidCallback(Callback):
    def __init__(self, num_classes, reserve_0=True, keys_and_classes=None,
                 fpi_thresholds=[0.5, 1.0, 2.0], iou=0.5, apply_nms=True,
                 warning_nms_gt=True, log_file=None):
        self.active = False
        self.num_classes = num_classes
        self.iou=iou
        self.reserve_0 = reserve_0
        if keys_and_classes is None:
            if reserve_0:
                keys_and_classes = {str(i): [i] for i in range(1, num_classes)}
            else:
                keys_and_classes = {str(i): [i] for i in range(num_classes)}
        self.keys_and_classes = keys_and_classes
        self.fpi_thresholds = fpi_thresholds
        self.reset()
        self.apply_nms = apply_nms
        self.warning_nms_gt = warning_nms_gt
        self.log_file = log_file

    def reset(self):
        self.gts_boxes = []
        self.gts_labels = []
        self.det_boxes = []
        self.det_labels = []
        self.det_scores = []

    def on_batch_ended(self, resources):
        if not self.active:
            return
        _, targets = resources["batch"]
        outs = resources["out"]
        assert len(targets) == len(outs)

        for target in targets:
            self.gts_boxes.append(copy.deepcopy(target["boxes"].detach().cpu().numpy()))
            self.gts_labels.append(copy.deepcopy(target["labels"].detach().cpu().numpy()))

        for out in outs:
            self.det_boxes.append(copy.deepcopy(out["boxes"].detach().cpu().numpy()))
            self.det_labels.append(copy.deepcopy(out["labels"].detach().cpu().numpy()))
            self.det_scores.append(copy.deepcopy(out["scores"].detach().cpu().numpy()))

    def on_train_started(self, resources):
        self.active = False

    def on_valid_started(self, resources):
        self.active = True
        self.reset()

    def on_valid_ended(self, resources):
        for key, classes in self.keys_and_classes.items():
            pred_pos_boxes = []
            pred_pos_scores = []
            for labels, boxes, scores in zip(self.det_labels, self.det_boxes, self.det_scores):
                bb = []
                sc = []
                for label, box, score in zip(labels, boxes, scores):
                    if label in classes:
                        bb.append(box)
                        sc.append(score)

                if len(bb)>0:
                    bb = np.array(bb)
                    sc = np.array(sc)
                    if self.apply_nms:
                        idx = nms(bb, sc, self.iou)
                        bb = bb[idx]
                        sc = sc[idx]
                pred_pos_boxes.append(bb)
                pred_pos_scores.append(sc)

            true_positive_boxes = []
            for labels, boxes in zip(self.gts_labels, self.gts_boxes):
                bb = []
                for label, box in zip(labels, boxes):
                    if label in classes:
                        bb.append(box)

                if len(bb)>0:
                    bb = np.array(bb)
                    if self.apply_nms:
                        idx = nms(bb, np.ones(bb.shape[0]), self.iou)
                        if self.warning_nms_gt and len(idx)<len(bb):
                            print("Supressed ground truth with non-maxima threshold")
                        bb = bb[idx]
                true_positive_boxes.append(bb)

            recalls = compute_recall_at_FPI(pred_pos_boxes,
                                            pred_pos_scores,
                                            true_positive_boxes,
                                            self.fpi_thresholds,
                                            self.iou)

            for fpi, recall in zip(self.fpi_thresholds, recalls):
                resources[f"epoch_valid_recall@{fpi}FPI_score (c={key})"] = recall

        """
        metrics = compute_detection_metrics((self.det_boxes, self.det_scores, self.det_labels),
                                            (self.gts_boxes, self.gts_labels),
                                             self.num_classes,
                                             self.iou)
        for i, ap in zip(range(self.num_classes), metrics["AP_per_class"]):
            resources["epoch_valid_AP_score (c={})".format(i)] = ap
        resources["epoch_valid_AP_score (average)"] = metrics["mAP"]

        for i, auc in zip(range(self.num_classes), metrics["rocauc_per_class"]):
            resources["epoch_valid_rocAUC_score (c={})".format(i)] = auc
        resources["epoch_valid_rocAUC_score (average)"] = metrics["mROCAUC"]

        for i, value in zip(range(self.num_classes), metrics["recall@FPI"]):
            resources["epoch_valid_recall@0.5FPI_score (c={})".format(i)] = value[0]
            resources["epoch_valid_recall@1.0FPI_score (c={})".format(i)] = value[1]
            resources["epoch_valid_recall@2.0FPI_score (c={})".format(i)] = value[2]
        resources["epoch_valid_recall@0.5FPI_score (average)"] = metrics["mrecall@FPI"][0]
        resources["epoch_valid_recall@1.0FPI_score (average)"] = metrics["mrecall@FPI"][1]
        resources["epoch_valid_recall@2.0FPI_score (average)"] = metrics["mrecall@FPI"][2]
        """
        self.log(resources)

    """
    def keys(self):
        keys=[]
        for i in range(self.num_classes):
            keys.append("epoch_valid_AP_score (c={})".format(i))
        keys.append("epoch_valid_AP_score (average)")
        for i in range(self.num_classes):
            keys.append("epoch_valid_rocAUC_score (c={})".format(i))
        keys.append("epoch_valid_rocAUC_score (average)")

        for fpi in "0.5", "1.0", "2.0":
            for i in range(self.num_classes):
                keys.append("epoch_valid_recall@{}FPI_score (c={})".format(fpi, i))
            keys.append("epoch_valid_recall@{}FPI_score (average)".format(fpi))
        return keys
    """

    def log(self, resources):
        print("\n")
        print("\t=== Detection Results ===")
        for key in self.keys_and_classes:
            text = f"\t {key} - Recall@FPI\n"
            for fpi in self.fpi_thresholds:
                text+= f"\t\t {fpi}"
            text += "\n"
            for fpi in self.fpi_thresholds:
                text += "\t\t {:.3f}".format(resources[f"epoch_valid_recall@{fpi}FPI_score (c={key})"])
            print(text)

        print("\t=========================")
        if self.log_file is not None:
            text = str(resources["epoch"])
            with open(self.log_file, "a") as file:
                for key in self.keys_and_classes:
                    for fpi in self.fpi_thresholds:
                        text += ", " + str(resources[f"epoch_valid_recall@{fpi}FPI_score (c={key})"])
                file.write(text + "\n")


"""
        for metric in ["AP", "rocAUC", "recall@0.5FPI", "recall@1.0FPI", "recall@2.0FPI"]:
            print(f"\t{metric}")
            keys = [f"(c={i})" for i in range(self.num_classes)] + ["(average)"]
            text = [f"{i}" for i in range(self.num_classes)] + ["(mean)"]
            print("\t"+"\t".join(text))
            scores = [f"epoch_valid_{metric}_score {k}" for k in keys]
            print("\t"+"\t".join([f"{resources[s]:.3f}" for s in scores]))
            print("\t\t-------------------------------------------------")
"""

def custom_pr_curve(y_true, y_prob, total_pos):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    args = np.argsort(y_prob)[::-1]
    y_true = y_true[args]
    y_prob = y_prob[args]

    TP = 0
    pred_pos = 0

    curve_recall = [0]
    curve_precision = [1]

    for i, (y, score) in enumerate(zip(y_true, y_prob)):
        if score < 0:
            break

        pred_pos += 1
        if y == 1:
            TP += 1

        if i<len(y_prob)-1:
            if score == y_prob[i+1]:
                continue

        recall = TP / total_pos
        precision = TP / pred_pos

        curve_recall.append(recall)
        curve_precision.append(precision)
    return curve_recall, curve_precision

def cummax(array):
    result = array.copy()
    curr_max = -np.inf
    for i, v in enumerate(array):
        curr_max = max(v, curr_max)
        result[i] = curr_max
    return result

def custom_average_precision(curve_recall, curve_precision):
    corrected_precisions = cummax(np.array(curve_precision)[::-1])[::-1]
    curr_recall = curve_recall[0]
    total = 0

    for recall, precision in zip(curve_recall, corrected_precisions):
        total += (recall-curr_recall) * precision
        curr_recall = recall
    return total
