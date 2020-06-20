# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) 

import os, time
import tkinter as tk
import tkinter.messagebox
from threading import Thread
from collections import deque

import numpy as np

from envs.crawler import CrawlerEnv
from .utils.plot import build_matplotlib_canvas, SurfacePlot

class App(tk.Frame):

    class Logger(object):

        def __init__(self, app):
            self.app = app
            self.gamma = None
            self.value_plt = SurfacePlot(122)

            self.velocity_plt_ax = self.value_plt.fig.add_subplot(121)
            self.velocity_plt_ax.relim()
            self.velocity_plt_ax.autoscale_view(True,True,True)
            self.velocity_plt, = self.velocity_plt_ax.plot([0], [0])
        
        def init(self):
            self.value_plt.init()
            self.value_plt.ax.set_xlabel("forearm")
            self.value_plt.ax.set_ylabel("upper arm")
            self.value_plt.ax.set_zlabel("value")
            self.value_plt.ax.invert_yaxis()
            self.value_plt.ax.view_init(azim=-135)
            self.velocity_plt_ax.set_xlabel("Iteration")
            self.velocity_plt_ax.set_ylabel("Average Speed")

        def log(self, step, v, pi=None, num_of_tests=10):
            self.draw_value(step, v)
            if pi is not None:
                self.draw_policy(pi)
                if self.model_based:
                    for _ in range(num_of_tests):
                        self.app.env.step(pi[self.app.env.state])

        def draw_value(self, step, v, sleep_time=0):
            self.value_plt.draw(
                np.reshape(v, (self.app.n_rows, self.app.n_cols)),
                "Iteration: {}".format(step), sleep_time
            )
        
        def draw_policy(self, pi, sleep_time=0):
            self.app.policy_canvas.delete("policy")
            for s, p in enumerate(pi):
                r = int(s / self.app.n_cols)
                c = s % self.app.n_cols
                self.app.draw_policy(r, c, p)
            time.sleep(sleep_time)
        
        def clear(self):
            self.velocity_plt_ax.clear()
            self.velocity_plt, = self.velocity_plt_ax.plot([], [])
            self.value_plt.clear()
            self.app.policy_canvas.delete("policy")
        
        def draw_velocity(self, step, vel):
            x = np.append(self.velocity_plt.get_xdata(), step)
            y = np.append(self.velocity_plt.get_ydata(), vel)
            self.velocity_plt.set_xdata(x)
            self.velocity_plt.set_ydata(y)
            self.velocity_plt_ax.relim()
            self.velocity_plt_ax.autoscale_view(True,True,True)
            # self.velocity_plt_ax.clear()
            # self.velocity_plt, = self.velocity_plt_ax.plot(x, y)
            self.velocity_plt_ax.set_xlabel("Iteration")
            self.velocity_plt_ax.set_ylabel("Average Reward")
            self.value_plt.fig.canvas.draw()
            self.value_plt.fig.canvas.flush_events()

    class Env(CrawlerEnv):
        def __init__(self, app):
            self.app = app
            self.velocity = deque()
            super().__init__()

            self.clemson_logo = tk.PhotoImage(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "clemson.png"))
            sx = int(self.clemson_logo.width()/(self.base_height*0.8))
            sy = int(self.clemson_logo.height()/(self.base_height*0.8))
            self.clemson_logo = self.clemson_logo.subsample(sx, sy)
        
        def reset(self):
            self.velocity.clear()
            self.vel_sum = 0
            self.app.logger.model_based = False
            return super().reset()

        def step(self, a):
            result = super().step(a)
            self.velocity.append(result[1])
            self.vel_sum += result[1]
            if len(self.velocity) > 120:
                x = self.velocity.popleft()
                self.vel_sum -= x
            self.app.logger.draw_velocity(self.steps, self.vel_sum/len(self.velocity))
            self.render()
            return result

        def render(self, canvas=None):
            h = self.app.world_canvas.winfo_height()-20
            self.app.world_canvas.delete("all")
            super().render(self.app.world_canvas)
            self.app.world_canvas.create_image(
                self.links["base"]["world_position"][0],
                h - self.links["base"]["world_position"][1],
                anchor=tk.CENTER, image=self.clemson_logo,
                tag="clemson"
            )
            self.app.world_canvas.update()



    def __init__(self, alg_fn_map, master=None):
        super().__init__(master)
        self.alg_fn_map = alg_fn_map

        self.logger = self.Logger(self)
        self.env = self.Env(self)

        self.master.title("Crawler -- CPSC 4820/6820 Clemson University")
        self.master.geometry("800x600")
        self.master.resizable(False, False)

        toolbar_frame = tk.Frame(master=self.master)
        plot_toolbar_frame = tk.Frame(toolbar_frame)
        plot_canvas, _ = build_matplotlib_canvas(
            self.logger.value_plt.fig, self.master, plot_toolbar_frame
        )
        self.plot_canvas = plot_canvas.get_tk_widget()
        self.logger.value_plt.fig.set_canvas(plot_canvas)
        self.logger.init()
        self.plot_canvas.config(height=300)

        self.world_canvas = tk.Canvas(self.master, bg="white", bd=0, highlightthickness=0, relief='ridge')
        self.policy_canvas = tk.Canvas(self.master, bg="white", bd=0, highlightthickness=0, relief='ridge')

        self.world_canvas.grid(row=0, column=0,
            sticky=tk.W+tk.E+tk.N+tk.S
        )
        self.policy_canvas.grid(row=0, column=1,
            sticky=tk.W+tk.E+tk.N+tk.S
        )
        self.plot_canvas.grid(row=1, column=0, columnspan=2,
            sticky=tk.W+tk.E+tk.N+tk.S
        )
        toolbar_frame.grid(row=2, column=0, columnspan=2,
            sticky=tk.W+tk.E, padx=10, pady=(10, 0)
        )

        alg_var = tk.StringVar(toolbar_frame)
        alg_var.set(next(iter(self.alg_fn_map.keys())))
        listbox_alg = tk.OptionMenu(toolbar_frame, alg_var, *self.alg_fn_map.keys())
        btn_solve = tk.Button(toolbar_frame, text="Solve")
        btn_test = tk.Button(toolbar_frame, text="Test", state=tk.DISABLED)

        plot_toolbar_frame.pack(side=tk.LEFT)
        btn_test.pack(side=tk.RIGHT, padx=10)
        btn_solve.pack(side=tk.RIGHT)
        listbox_alg.pack(side=tk.RIGHT)

        def stop_request():
            self.stop_request = True

        def test(pi):
            listbox_alg.config(state=tk.DISABLED)
            btn_solve.config(state=tk.DISABLED)
            btn_test.config(text="Stop")
            btn_test.config(state=tk.NORMAL)
            self.stop_request = False
            btn_test.config(command=stop_request)
            # def test_thread():
            self.test_policy(pi)
            listbox_alg.config(state=tk.NORMAL)
            btn_solve.config(state=tk.NORMAL)
            btn_test.config(text="Test")
            btn_test.config(state=tk.NORMAL)
            btn_test.config(command=lambda: test(pi))
            # Thread(target=test_thread).start()

        def solve():
            listbox_alg.config(state=tk.DISABLED)
            btn_solve.config(state=tk.DISABLED)
            btn_test.config(state=tk.DISABLED)
            btn_test.config(command=lambda: None)
            self.env.reset()
            self.logger.model_based = True
            # We assume the alg is model-based
            # such that auto test will be performed
            # It will be reset to False in self.env.reset()
            # such that in the user's alg, reset() is called (for model-free cases)
            # auto test will not be performed
            self.logger.clear()
            solver = self.alg_fn_map[alg_var.get()]
            pi = solver(self.env, self.logger)
            if pi is None:
                listbox_alg.config(state=tk.NORMAL)
                btn_solve.config(state=tk.NORMAL)
                tk.messagebox.showinfo("", "No policy received.")
            else:
                valid = True
                for s in range(self.env.observation_space.n):
                    if s not in pi or pi[s] not in range(self.env.action_space.n):
                        valid = False
                    if not valid:
                        break
                if valid:
                    listbox_alg.config(state=tk.NORMAL)
                    btn_solve.config(state=tk.NORMAL)
                    tk.messagebox.showinfo("", "Invalid policy received.")
                else:
                    listbox_alg.config(state=tk.NORMAL)
                    btn_solve.config(state=tk.NORMAL)
                    btn_test.config(state=tk.NORMAL)
                    btn_test.config(command=lambda: test(pi))
                    tk.messagebox.showinfo("", "Optimization is done.")
        
        btn_solve.config(command=solve)
        
        self.master.grid_columnconfigure(0, weight=3)
        self.master.grid_columnconfigure(1, weight=2)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=0)

        self.world_canvas.bind("<Configure>", lambda _: self.env.render(self.world_canvas) and False or self.draw_policy_grid())

    def draw_policy_grid(self):
        self.policy_canvas.delete("grid")
        margin = (20, 40)
        w = self.policy_canvas.winfo_width()-margin[0]
        h = self.policy_canvas.winfo_height()-margin[1]

        self.n_cols = self.env.joints["forearm"]["space"]+1
        self.n_rows = self.env.joints["upper_arm"]["space"]+1
        self.cell_size = min(w/self.n_cols, h/self.n_rows)
        self.board_pos = (
            (w - self.cell_size*self.n_cols + margin[0])*0.5,
            (h - self.cell_size*self.n_rows + margin[1])*0.5,
            (w + self.cell_size*self.n_cols + margin[0])*0.5,
            (h + self.cell_size*self.n_rows + margin[1])*0.5,
        )

        # self.policy_canvas.create_text(
        #     self.board_pos[0]-10, (self.board_pos[1]+self.board_pos[3])*0.5,
        #     text="upper arm", font=(None, 10),
        #     anchor=tk.E, angle=90
        # )
        # self.policy_canvas.create_text(
        #     (self.board_pos[0]+self.board_pos[2])*0.5, self.board_pos[3], 
        #     text="forearm", font=(None, 10),
        #     anchor=tk.N
        # )
        for r in range(self.n_rows+1):
            self.policy_canvas.create_line(
                self.board_pos[0], self.board_pos[1] + r*self.cell_size,
                self.board_pos[2], self.board_pos[1] + r*self.cell_size,
                fill="gray", tag="grid"
            )
        for c in range(self.n_cols+1):
            self.policy_canvas.create_line(
                self.board_pos[0] + c*self.cell_size, self.board_pos[1],
                self.board_pos[0] + c*self.cell_size, self.board_pos[3],
                fill="gray", tag="grid"
            )
    
    def draw_policy(self, r, c, a):
        r += 0.5
        c += 0.5
        if a == 0:      # upper arm - 1
            r_, c_ = r-0.5, c
        elif a == 1:    # forearm - 1
            r_, c_ = r, c-0.5
        elif a == 2:    # upper arm + 1
            r_, c_ = r+0.5, c
        else:           # forearm + 1
            r_, c_ = r, c+0.5

        self.policy_canvas.create_line(
            self.board_pos[0]+self.cell_size*c,  self.board_pos[1]+self.cell_size*r,
            self.board_pos[0]+self.cell_size*c_, self.board_pos[1]+self.cell_size*r_,
            fill="gray25",
            arrow=tk.LAST, tag="policy"
        )

    def test_policy(self, pi, steps=120):
        for _ in range(steps):
            if self.stop_request:
               break 
            self.env.step(pi[self.env.state])
    
        