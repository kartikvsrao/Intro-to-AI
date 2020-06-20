# regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

import enum
import tkinter as tk
import tkinter.messagebox
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
def build_matplotlib_canvas(figure, canvas_master, toolbar_master=None):
    plot_canvas = FigureCanvasTkAgg(figure, master=canvas_master)
    plot_toolbar = None if toolbar_master is None else NavigationToolbar2Tk(plot_canvas, toolbar_master)
    plot_canvas.mpl_connect("key_press_event",
        lambda e: matplotlib.backend_bases.key_press_handler(e, plot_canvas, plot_toolbar)
    )
    return plot_canvas, plot_toolbar


def confusion_matrix(pred, ground, classes=None):
    if classes is None:
        classes = np.sort(np.union1d(np.unique(pred), np.unique(ground)))
    n_classes = len(classes)
    class_map = {k: v for v, k in enumerate(classes)}
    mat = np.zeros((n_classes, n_classes), dtype=np.int32)
    for p, g in zip(pred, ground):
        i = class_map[p]
        j = class_map[g]
        mat[i][j] += 1
    return mat, classes

def pca(data, n_components):
    n = len(data)
    x = np.subtract(data, np.mean(data, axis=0))
    _, _, v = np.linalg.svd(x, full_matrices=False)
    v = np.transpose(v)[:, :n_components]
    return lambda _: np.dot(_, v)


class App(tk.Frame):

    class TaskType(enum.Enum):
        REGRESSION = 0
        CLASSIFICATION = 1
        BINARY_CLASSIFICATION = 2

    class Logger(object):

        def __init__(self, app):
            self.fig = matplotlib.figure.Figure()
        
        def init(self):
            matplotlib.pyplot.ion()
            self.training_ax = self.fig.add_subplot(211)
            self.testing_ax = self.fig.add_subplot(212)
            self.training_plot, = self.training_ax.plot([], [])
            self.training_ax.set_ylabel("Loss")
            self.testing_plot = None
        
        def clear(self):
            self.training_ax.clear()
            self.testing_ax.clear()
            self.training_plot, = self.training_ax.plot([], [])
            self.training_ax.set_ylabel("Loss")
            self.testing_plot = None
        
        def set_title(self, title):
            self.fig.suptitle(title, fontsize=12)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        def log(self, step, loss):
            x = np.append(self.training_plot.get_xdata(), step)
            y = np.append(self.training_plot.get_ydata(), loss)
            self.training_plot.set_xdata(x)
            self.training_plot.set_ydata(y)
            self.training_ax.relim()
            self.training_ax.autoscale_view(True,True,True)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            

    def __init__(self, dbs, algs, master=None):
        super().__init__(master)
        self.dbs = dbs
        self.algs = algs

        self.master.title("Regression -- CPSC 4820/6820 Clemson University")
        # self.master.geometry("800x600")
        self.master.resizable(True, True)
        
        self.logger = self.Logger(self)
        plot_toolbar_frame = tk.Frame(master=self.master)
        training_canvas, _ = build_matplotlib_canvas(
            self.logger.fig, self.master, plot_toolbar_frame
        )
        self.logger.fig.set_canvas(training_canvas)
        self.logger.init()

        self.btn_solve = tk.Button(self.master, text="Solve", command=self.solve)
        training_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=1
        )
        plot_toolbar_frame.pack(
            side=tk.LEFT, pady=(10, 0)
        )
        self.btn_solve.pack(
            side=tk.RIGHT, pady=(10, 0)
        )
    
        self.solve_window = None


    def solve(self):
        if self.solve_window is not None:
            self.solve_window.update()
            self.solve_window.deiconify()
            return
        
        self.solve_window = tk.Toplevel(self.master)
        self.solve_window.title("Solver Options")
        self.solve_window.protocol("WM_DELETE_WINDOW", self.solve_window.withdraw)
        self.solve_window.resizable(False, False)

        db_var = tk.StringVar(self.solve_window)
        db_var.set(next(iter(self.dbs.keys())))
        listbox_db = tk.OptionMenu(self.solve_window, db_var, *self.dbs.keys())

        def threshold_validate(text, value_if_allowed):
            if len(value_if_allowed) > 0:
                try:
                    v = int(value_if_allowed)
                except ValueError:
                    return False
            return v < 100 and v > 0
        threshold_var = tk.StringVar(self.solve_window)
        threshold_var.set("50")
        field_threshold = tk.Entry(self.solve_window, width=3,
            textvariable=threshold_var,
            validate="key"
        )
        alg_var = tk.StringVar(self.solve_window)
        alg_var.set(next(iter(self.algs.keys())))
        listbox_alg = tk.OptionMenu(self.solve_window, alg_var, *self.algs.keys())

        def need_threshold():
             return False
             #return self.dbs[db_var.get()][1] == self.TaskType.REGRESSION and \
             #   self.algs[alg_var.get()][1] != self.TaskType.REGRESSION 
            
        db_var.trace("w",
            lambda *db: field_threshold.grid(row=0, column=2, padx=2) \
                if need_threshold() else field_threshold.grid_forget()
        ) 
        alg_var.trace("w",
            lambda *db: field_threshold.grid(row=0, column=2, padx=2) \
                if need_threshold() else field_threshold.grid_forget()
        )

        def solve():
            if need_threshold():
             #   tk.messagebox.showinfo("", "Please use an appropriate dataset for classification problems")
             #   return
                failed = False
                try:
                    threshold = int(threshold_var.get())
                except ValueError:
                    failed = True
                if failed or threshold < 0 or threshold >= 100:
                    tk.messagebox.showinfo("", "Threshold to binarize the dataset must be an integer between 0 and 100 (excluded).")
                    return
            
            db = self.dbs[db_var.get()]
            solver = self.algs[alg_var.get()]
            if solver[1] == self.TaskType.BINARY_CLASSIFICATION and db[1] == self.TaskType.CLASSIFICATION:
                tk.messagebox.showinfo("", "Binary classification solver, {}, cannot solve the multi-class problem {}.".format(alg_var.get(), db_var.get()))
                return
            if solver[1] == self.TaskType.BINARY_CLASSIFICATION and db[1] == self.TaskType.REGRESSION:
                tk.messagebox.showinfo("", "A binary classification solver should not be applied to the {} dataset.".format(db_var.get()))
                return
     
            btn_submit.config(state=tk.DISABLED)
            self.btn_solve.config(state=tk.DISABLED)
            self.solve_window.withdraw()

            self.logger.clear()
            if need_threshold():
                db_name = db_var.get() + " " + str(threshold)
                pass
            else:
                db_name = db_var.get()
            self.logger.set_title(db_name + "\n" + alg_var.get())

            x, y = db[0]()
            x, y = np.array(x), np.array(y)
            if need_threshold():
                t = np.quantile(y, threshold/100.0)
                p = np.where(y >= t)
                y_ = np.zeros(y.shape, dtype=np.int32)
                y_[p] = 1
                y = y_
            elif solver[1] != self.TaskType.REGRESSION:
                y = y.astype(np.int32)
            
            x = (x - np.mean(x, axis=0, keepdims=True)) / np.maximum(1e-6, np.std(x, axis=0, keepdims=True))
            x = np.insert(x, 0, 1, axis=1)
            if need_threshold() or solver[1] != self.TaskType.REGRESSION:
                x_train, y_train = None, None
                x_test, y_test = None, None

                for c in np.unique(y):
                    idx = np.where(y == c)[0]
                    np.random.shuffle(idx)
                    n = len(idx)
                    n_train = int(n*0.8)
                    if x_train is None:
                        x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
                        x_test, y_test = x[idx[n_train:]], y[idx[n_train:]]
                    else:
                        x_train = np.concatenate((x_train, x[idx[:n_train]]), axis=0)
                        y_train = np.append(y_train, y[idx[:n_train]])
                        x_test = np.concatenate((x_test, x[idx[n_train:]]), axis=0)
                        y_test = np.append(y_test, y[idx[n_train:]])
                idx = np.arange(len(x_train))
                np.random.shuffle(idx)
                x_train, y_train = x_train[idx], y_train[idx]
            else:
                n = len(x)
                idx = np.arange(n)
                np.random.shuffle(idx)
                n_train = int(n*0.8)
                x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
                x_test, y_test = x[idx[n_train:]], y[idx[n_train:]]
            task_type = solver[1]

            w = solver[0](x_train.tolist(), y_train.tolist(), self.logger)
            self.summarize(task_type, x_train, y_train, x_test, y_test, w)

            self.btn_solve.config(state=tk.NORMAL)
            btn_submit.config(state=tk.NORMAL)

            
        btn_submit = tk.Button(self.solve_window, text="Done", command=solve)

        tk.Label(self.solve_window, text="Dataset").grid(
            row=0, column=0, padx=10, sticky=tk.E
        )
        listbox_db.grid(
            row=0, column=1, padx=10, sticky=tk.W
        )
        if need_threshold():
            field_threshold.grid(
                row=0, column=2, padx=2,
            )
        tk.Label(self.solve_window, text="Algorithm").grid(
            row=1, column=0, padx=10, sticky=tk.E
        )
        listbox_alg.grid(
            row=1, column=1, columnspan=2, padx=10, sticky=tk.W
        )

        btn_submit.grid(
            row=2, columnspan=3, pady=(20, 0)
        )

    def summarize(self, task_type, x_train, y_train, x_test, y_test, w):
        if task_type == self.TaskType.REGRESSION:
            try:
                pred_train = np.matmul(x_train, w)
                pred_test = np.matmul(x_test, w)
                train_err = 0.5 / len(y_train) * np.sum(np.square(pred_train - y_train))
                test_err = 0.5 / len(y_test) * np.sum(np.square(pred_test - y_test))
            except:
                tk.messagebox.showinfo("", "Invalid data received from solver function.")
                return
            self.logger.testing_ax.scatter(y_train, pred_train, s=10, alpha=0.5, label="Train Error: {:.4f}".format(train_err))
            self.logger.testing_ax.scatter(y_test, pred_test, s=10, alpha=0.5, label="Test Error: {:.4f}".format(test_err))
            diag_line, = self.logger.testing_ax.plot([], [], ls="--", alpha=0.2, c="black")
            self.logger.testing_ax.callbacks.connect("xlim_changed", lambda ax: diag_line.set_data(ax.get_xlim(), ax.get_ylim()))
            self.logger.testing_ax.callbacks.connect("ylim_changed", lambda ax: diag_line.set_data(ax.get_xlim(), ax.get_ylim()))
            self.logger.testing_ax.legend(fontsize=8)
            self.logger.testing_ax.set_aspect("equal", "datalim")
        else:
            try:
                if task_type == self.TaskType.BINARY_CLASSIFICATION:
                    pred = np.matmul(x_train, w)
                    pred_train = np.zeros(pred.shape, dtype=y_train.dtype)
                    pred_train[np.where(pred >= 0)] = 1
                    pred = np.matmul(x_test, w)
                    pred_test = np.zeros(pred.shape, dtype=y_test.dtype)
                    pred_test[np.where(pred >= 0)] = 1
                else:
                    pred_train = np.argmax(np.matmul(x_train, w), axis=1)
                    pred_test = np.argmax(np.matmul(x_test, w), axis=1)
            except:
                tk.messagebox.showinfo("", "Invalid data received from solver function.")
                return
            
            cm_train, class_map = confusion_matrix(pred_train, y_train)
            cm_test, class_map = confusion_matrix(pred_test, y_test, class_map)
            print('Training accuracy: %.3f' % (np.trace(cm_train)/np.sum(cm_train)))
            print('Testing accuracy: %.3f' % (np.trace(cm_test)/np.sum(cm_test)))
            im = self.logger.testing_ax.imshow(cm_test, interpolation='nearest', cmap=matplotlib.cm.GnBu)
            self.logger.testing_ax.set_title('Train/Test confusion matrix', fontsize=12)
            thresh = cm_test.max() / 2
            for i in range(cm_test.shape[0]):
                for j in range(cm_test.shape[1]):
                    self.logger.testing_ax.text(j, i, "{}/{}".format(cm_train[i,j], cm_test[i,j]), fontsize=8,
                                ha="center", va="center",
                                color="white" if cm_test[i, j] > thresh else "black")
            #self.logger.testing_ax.text(
            #    self.logger.testing_ax.get_xlim()[0], self.logger.testing_ax.get_ylim()[0],
            #    "train / test", fontsize=8, ha="right", va="top", bbox={'facecolor': 'white', 'pad': 6})
            
            self.logger.testing_ax.set_yticks(list(range(cm_test.shape[0])))
            self.logger.testing_ax.set_xticks(list(range(cm_test.shape[1])))
            self.logger.testing_ax.set_yticklabels(list(range(cm_test.shape[0])))
            self.logger.testing_ax.set_xticklabels(list(range(cm_test.shape[1])))
            self.logger.testing_ax.set_aspect("auto")

            x = np.concatenate([x_train, x_test], axis=0)
            x = pca(x, n_components=2)(x)
            x_train = x[:len(x_train)] #pca.transform(x_train)
            x_test = x[len(x_train):] #pca.transform(x_test)

            self.logger.training_ax.clear()
            for i in np.unique(np.concatenate((y_train, y_test))):
                idx = np.where(np.equal(y_train, i) & np.equal(pred_train, i))
                p = self.logger.training_ax.scatter(x_train[idx, 0], x_train[idx, 1], s=10, alpha=0.5, marker="o", label="{}".format(i))
                c = p.get_facecolor()
                idx = np.where(np.equal(y_train, i) & np.not_equal(pred_train, i))
                self.logger.training_ax.scatter(x_train[idx, 0], x_train[idx, 1], s=10, alpha=0.5, c=c, marker="x")
                idx = np.where(np.equal(y_test, i) & np.equal(pred_test, i))
                self.logger.training_ax.scatter(x_test[idx, 0], x_test[idx, 1], s=10, alpha=0.5, c=c, marker="s")
                idx = np.where(np.equal(y_test, i) & np.not_equal(pred_test, i))
                self.logger.training_ax.scatter(x_test[idx, 0], x_test[idx, 1], s=10, alpha=0.5, c=c, marker="+")
            
            self.logger.training_ax.text(
                self.logger.training_ax.get_xlim()[0], self.logger.training_ax.get_ylim()[1],
                "train accuracy: {:.2f}, test accuracy: {:.2f}".format(sum(np.diag(cm_train))/len(x_train), sum(np.diag(cm_test))/len(x_test)),
                fontsize=8, ha="left", va="top",
                bbox={'facecolor': 'white', 'pad': 7})
            self.logger.training_ax.legend(fontsize=8)
            self.logger.training_ax.set_yticks([])
            self.logger.training_ax.set_xticks([])

        self.logger.testing_ax.set_xlabel("True Value")
        self.logger.testing_ax.set_ylabel("Predicted Value")
        self.logger.fig.canvas.draw()
        self.logger.fig.canvas.flush_events()
