from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import pickle as pkl

if __name__=="__main__":    
    
    # with open("/media/emcastro/TOSHIBA EXT/Eduardo/final results/architecture/second table/results_small.txt", "r") as file:
    with open("/home/emcastro/deepmm/results_small.txt", "r") as file:
        lines = file.readlines()
    for i, name in [(1, "Accuracy"), (3, "Balanced Accuracy"), (5, "rocAUC"), (7, "F1-score")]:
        plt.close("all")
        xs = [x for x in range(len(lines))]
        ys = [float(x.split(",")[i]) for x in lines]
        errs = [float(x.split(",")[i+1]) for x in lines]
        xlabels = ["z2", "p4-1", "p4-2", "p4-3", "p4-4", "p4", "p4"]
        ylabel = name
        xlabel = "Architecture"
        minl = np.min(ys) -.01
        maxl = np.max(ys) +.01
        ylim = (minl,maxl)
        
        d = {"ylabel": ylabel,
             "xlabel": xlabel,
             "xs":xs,
             "ys":ys,
             "errs":errs,
             "xlabels":xlabels,
             "ylim": ylim}
    
        path = "/home/emcastro/bar_plot_data.pkl"
    
        with open(path, "wb") as file:
            pkl.dump(d, file)
        
        path = "/home/emcastro/bar_plot_data.pkl"
        
        with open(path, "rb") as file:
            plot_data = pkl.load(file)
        
        xlabel = plot_data["xlabel"]
        xs = np.array(plot_data["xs"])
        ys = np.array(plot_data["ys"])
        
        if "xlabels" in plot_data.keys():
            plt.xticks(xs, plot_data["xlabels"])
            
        if "xlabel" in plot_data.keys():
            plt.xlabel(plot_data["xlabel"])
    
        if "ylabel" in plot_data.keys():
            plt.ylabel(plot_data["ylabel"])
    
        if "ylim" in plot_data.keys():
            plt.ylim(*plot_data["ylim"])
            
        if "title" in plot_data.keys():
            plt.title(plot_data["title"])
        
        if "errs" in plot_data.keys():
            errs = np.array(plot_data["errs"])
            error_kw = {'capsize': 5, 'capthick': 1, 'ecolor': 'black'}
        else:
            errs = None
            error_kw =None
    
        with plt.style.context(['science']):
            if errs is not None:
                plt.bar(xs, ys, yerr=errs, error_kw=error_kw)
            else:
                plt.bar(xs, ys)
            plt.show()
            
        plt.tight_layout()
        plt.savefig(f"/home/emcastro/{name}.pdf")
