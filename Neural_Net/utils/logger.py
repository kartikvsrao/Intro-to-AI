import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Logger(object):

    def __init__(self, max_epochs, db):
        plt.ion()
        matplotlib.rcParams['font.size'] = 8
        with plt.style.context("seaborn-paper"):
            matplotlib.rcParams['font.size'] = 8
            self.fig = plt.figure(figsize=( 6,3)  , dpi=100)
            gs = matplotlib.gridspec.GridSpec(1, 2)
            self.loss_ax = self.fig.add_subplot(gs[0,0])
            self.loss_ax.plot([], [], label="Train Loss", lw=0.8)
            self.loss_ax.plot([], [], label="Val Loss", lw=0.8)
            self.loss_ax.legend()
            self.loss_ax.grid(True, alpha=0.3)
            self.acc_ax = self.fig.add_subplot(gs[0,1])
            self.acc_ax.plot([], [], label="Train Accuracy", lw=0.8)
            self.acc_ax.plot([], [], label="Val Accuracy", lw=0.8)
            self.acc_ax.legend()
            self.acc_ax.grid(True, alpha=0.3)
        
        self.max_epochs = max_epochs
        self.db = db
        self.step = 0
        self.start_time = None
        self.train_loss = 0
        self.train_acc = 0
        self.val_loss = 0
        self.val_acc = 0
        self.loss_sum = 0
        self.acc_sum = 0
        self.global_step = 0
    
    def start(self, epoch):
        self.epoch = epoch
        self.start_time = time.time()
        self.step = 1
        print("Epoch: {}/{}".format(self.epoch, self.max_epochs))
    
    def log(self, loss, acc, global_step, plot=False):
        eta = time.time() - self.start_time
        self.step += 1
        self.loss_sum += loss
        self.acc_sum += acc
        self.train_loss = self.loss_sum / self.step
        self.train_acc = self.acc_sum / self.step
        self.global_step = global_step

        progress = min(11, 12* self.db.train.cursor//self.db.train.n_examples)
        remaining = 11 - progress
        print("  {}/{} [={}{}] - ETA: {:d}s - loss: {:.4f} - acc: {:.4f}".format(
            str(self.db.train.cursor).rjust(len(str(self.db.train.n_examples))), self.db.train.n_examples,
            "="*progress, " "*remaining, int(eta), 
            self.train_loss, self.train_acc
        ), end="\r")


        if plot:
            self.fig.suptitle("Epoch: {}, Step: {}".format(self.epoch, self.step))
            train_loss_line = self.loss_ax.lines[0]
            x = np.append(train_loss_line.get_xdata(), global_step)
            y = np.append(train_loss_line.get_ydata(), loss)
            train_loss_line.set_xdata(x)
            train_loss_line.set_ydata(y)
            self.loss_ax.legend([
                "Train Loss: {:.4f}".format(self.train_loss),
                "Val Loss: {:.4f}".format(self.val_loss)
            ], loc=2)
            self.loss_ax.relim()
            self.loss_ax.autoscale_view(True,True,True)
            train_acc_line = self.acc_ax.lines[0]
            x = np.append(train_acc_line.get_xdata(), global_step)
            y = np.append(train_acc_line.get_ydata(), acc)
            train_acc_line.set_xdata(x)
            train_acc_line.set_ydata(y)
            self.acc_ax.legend([
                "Train Accuracy: {:.4f}".format(self.train_acc),
                "Val Accuracy: {:.4f}".format(self.val_acc)
            ], loc=3)
            self.acc_ax.relim()
            self.acc_ax.autoscale_view(True,True,True)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def validate(self, loss, acc):
        self.val_acc = acc
        self.val_loss = loss
        if self.start_time is not None:
            eta = time.time() - self.start_time
            progress = min(11, 12* self.db.train.cursor//self.db.train.n_examples)
            remaining = 11 - progress
            print("  {}/{} [={}{}] - ETA: {:d}s - loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}".format(
                str(self.db.train.cursor).rjust(len(str(self.db.train.n_examples))), self.db.train.n_examples,
                "="*progress, " "*remaining, int(eta), 
                self.train_loss, self.train_acc, self.val_loss, self.val_acc
            ), end="\r")
        
        val_loss_line = self.loss_ax.lines[1]
        x = np.append(val_loss_line.get_xdata(), self.global_step)
        y = np.append(val_loss_line.get_ydata(), loss)
        val_loss_line.set_xdata(x)
        val_loss_line.set_ydata(y)
        self.loss_ax.legend([
            "Train Loss: {:.4f}".format(self.train_loss),
            "Val Loss: {:.4f}".format(self.val_loss)
        ], loc=2)
        self.loss_ax.relim()
        self.loss_ax.autoscale_view(True,True,True)
        val_acc_line = self.acc_ax.lines[1]
        x = np.append(val_acc_line.get_xdata(), self.global_step)
        y = np.append(val_acc_line.get_ydata(), acc)
        val_acc_line.set_xdata(x)
        val_acc_line.set_ydata(y)
        self.acc_ax.legend([
            "Train Accuracy: {:.4f}".format(self.train_acc),
            "Val Accuracy: {:.4f}".format(self.val_acc)
        ], loc=3)
        self.acc_ax.relim()
        self.acc_ax.autoscale_view(True,True,True)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def done(self):
        self.start_time = None
        self.loss_sum = 0
        self.acc_sum = 0
        print()
    
    def confusion_matrix_for_test_set(self, probs, error_gallery=True):
        if error_gallery:
            fig = plt.figure()
            gs = matplotlib.gridspec.GridSpec(self.db.n_classes, 5)
            axes = [fig.add_subplot(gs[i,:]) for i in range(self.db.n_classes)]

        correct = 0
        confusion_mat = np.zeros((self.db.n_classes, self.db.n_classes), dtype=np.int32)
        for x, y, p in zip(self.db.test.data, self.db.test.labels, probs):
            i = np.argmax(p)
            confusion_mat[i][y] += 1
            if i == y:
                correct += 1
            elif error_gallery:
                if x.shape[0] == 1:
                    x = np.squeeze(x, 0)
                else:
                    np.transpose(x, [1, 2, 0])

                ax = axes[y]
                col = p[y]*10 - 0.5
                ax.imshow(x, vmin=0, vmax=255, alpha=0.3,
                    cmap="Reds",
                    extent=(col*28, col*28+28, 0, 28))
                ax.set_xlim([0, 140])
                ax.set_ylim([0, 28])
                ax.set_xticks([])
                ax.set_yticks([])
        acc = correct / self.db.test.n_examples
        if error_gallery:
            axes[-1].set_xticks([28*i for i in range(6)])
            axes[-1].set_xticklabels(["{:.1f}".format(0.1*i) for i in range(6)])
            axes[-1].set_xlabel("Probability of Ground Truth Label")
            fig.suptitle("Test Error Rate: {:.4f}".format(1-acc))
        print("Test Accuracy: {:.4f}".format(acc))

        fig, ax = plt.subplots(figsize=(self.db.n_classes, self.db.n_classes))
        ax.imshow(confusion_mat, interpolation='nearest', cmap=matplotlib.cm.GnBu)
        medium = confusion_mat.max() / 2
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                ax.text(j, i, "{}".format(confusion_mat[i,j]), fontsize=8,
                    ha="center", va="center",
                    color="white" if confusion_mat[i, j] > medium else "black")
        ax.set_xticks([i for i in range(self.db.n_classes)])
        ax.set_yticks([i for i in range(self.db.n_classes)])
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")

        plt.show(block=True)
