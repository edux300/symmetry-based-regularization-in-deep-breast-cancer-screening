from matplotlib import pyplot as plt
import numpy as np
from glob import glob

def process(results=None):
    N = 15
    xs = np.arange(N) * 5
    means = (np.arange(N)/N)**0.25
    stds = np.random.rand(N)*0.1+0.1
    return xs, means, stds

def draw(xs, means, stds, label=""):
    with plt.style.context(['science', 'vibrant']):

        limits = xs.min(), xs.max()
        plt.fill_between(xs, means-stds, means+stds, alpha=0.25)
        plt.plot(xs, means, label=label)

        plt.show()
        plt.xlim(*limits)

if __name__=="__main__":
    """
    path = ""
    #path = "/run/user/1000/gvfs/sftp:host=94.132.51.18,user=emcastro"
    path+= "/media/emcastro/External_drive/results/Works_on_Regularization/"

    means = []
    stds = []
    xs = []

    for i in range(5, 80, 5):
        scores = []
        for j in range(1, 5):
            mpath = path+f"CBIS_mass_p4_64_scratch_AdamW_custom_{i}_base_{j}"
            models = glob(mpath+"/model_average_valid_bal-accuracy*.pth")
            if len(models)==0:
                print(i, j)
                continue
            models = [float(m.split(".pth")[0].split("_")[-1]) for m in models]
            score = max(models)
            scores.append(score)
        means.append(np.mean(scores))
        stds.append(np.std(scores))
        xs.append(i)
    """

    xlabel = "Pre-training epochs"
    label = "p4 width=64"
    baseline_label = "z2 baseline"

    metric = "auc"

    if metric == "acc":
        # accuracy
        xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        ylabel = "Accuracy"
        means = [0.84564, 0.88362, 0.87904, 0.88452, 0.88362, 0.87948, 0.88538, 0.8799, 0.88298, 0.88014, 0.88582, 0.88384, 0.8843, 0.88778, 0.88866, 0.88712, 0.8847]
        stds = [0.00253, 0.00584, 0.00563, 0.00670, 0.00308, 0.00444, 0.00382, 0.00364, 0.00374, 0.00547, 0.00624, 0.00601, 0.00526, 0.00666, 0.00416, 0.00285, 0.00246]
        baseline_means = [0.88036] * len(xs)
        baseline_stds = [0.002018167486] * len(xs)
        limits = [.7, .9]
    elif metric == "bal":
        # bal-accuracy
        xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        ylabel = "Bal-Accuracy"
        means = [0.75824, 0.82104, 0.80296, 0.80936, 0.82284, 0.8085, 0.81618, 0.8111, 0.82084, 0.81728, 0.82072, 0.81186, 0.82248, 0.82112, 0.82564, 0.8185, 0.81912]
        stds = [0.01517, 0.00732, 0.00959, 0.00489, 0.00805, 0.00484, 0.00794, 0.00776, 0.00450, 0.00515, 0.00448, 0.00900, 0.00563, 0.00676, 0.00894, 0.00373, 0.00778]
        baseline_means = [0.82254] * len(xs)
        baseline_stds = [0.01327546609] * len(xs)
        limits = [.7, .9]
    elif metric == "auc":
        # rocAUC
        xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        ylabel = "roc AUC"
        means = [0.90406, 0.93792, 0.93488, 0.93292, 0.93864, 0.92972, 0.93498, 0.93642, 0.93678, 0.9383, 0.93722, 0.93804, 0.9393, 0.93962, 0.93872, 0.93766, 0.93728]
        stds = [0.00397, 0.00290, 0.00404, 0.00323, 0.00172, 0.00350, 0.00261, 0.00223, 0.00293, 0.00377, 0.00114, 0.00344, 0.00261, 0.00178, 0.00259, 0.00344, 0.00183]
        baseline_means = [0.93996] * len(xs)
        baseline_stds = [0.001976866207] * len(xs)
        limits = [.9, 1]
    elif metric == "f1":
        # f1-score
        xs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        ylabel = "F1-score"
        means = [0.75162, 0.81302, 0.80578, 0.81074, 0.81726, 0.80906, 0.819, 0.80856, 0.81772, 0.81268, 0.81878, 0.81782, 0.81864, 0.82476, 0.8226, 0.81866, 0.82008]
        stds = [0.00924, 0.00827, 0.00918, 0.00502, 0.00423, 0.00728, 0.00451, 0.00549, 0.00582, 0.01067, 0.00687, 0.00796, 0.01062, 0.01003, 0.00778, 0.00659, 0.00463]
        baseline_means = [0.81912] * len(xs)
        baseline_stds = [0.006113264267] * len(xs)
        limits = [.7, .9]

    means = np.array(means)
    stds = np.array(stds)
    xs = np.array(xs)

    draw(xs, means, stds, label=label)


    baseline_means = np.array(baseline_means)
    baseline_stds = np.array(baseline_stds)

    draw(xs, baseline_means, baseline_stds, label=baseline_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(*limits)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/home/emcastro/{metric}.pdf")