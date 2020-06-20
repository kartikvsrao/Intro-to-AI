# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas

import os
import time
import math
import tkinter as tk
import tkinter.messagebox
from threading import Thread

import numpy as np

from envs.grid_world import GridWorld
from .utils.plot import build_matplotlib_canvas, SurfacePlot


class App(tk.Frame):

    class Logger(object):

        def __init__(self, app):
            self.app = app
            self.gamma = None
            self.value_plt = SurfacePlot(211)

            self.test_plt_ax = self.value_plt.fig.add_subplot(212)
            self.test_plt_ax.relim()
            self.test_plt_ax.autoscale_view(True,True,True)
            self.test_plt, = self.test_plt_ax.plot([0], [0])
        
        def init(self):
            self.value_plt.init()
            self.value_plt.ax.set_xlabel("cols")
            self.value_plt.ax.set_ylabel("rows")
            self.value_plt.ax.set_zlabel("value")
            self.value_plt.ax.invert_yaxis()
            self.value_plt.ax.view_init(azim=-135)
            self.test_plt_ax.set_xlabel("Iteration")
            self.test_plt_ax.set_ylabel("Accumulated Reward")

        def log(self, step, v, pi=None, num_of_tests=10):
            self.draw_value(step, v)
            if pi is not None:
                self.draw_policy(pi)
                if num_of_tests > 0:
                    self.test_policy(step, pi, num_of_tests)

        def draw_value(self, step, v, sleep_time=0):
            self.value_plt.draw(
                np.reshape(v, (self.app.env.n_rows, self.app.env.n_cols)),
                "Iteration: {}".format(step), sleep_time
            )
        
        def draw_policy(self, pi, title=None, sleep_time=0):
            self.app.world_canvas.delete("policy")
            for s, p in enumerate(pi):
                if s in self.app.env.goals or s in self.app.env.obstacles:
                    continue
                r = int(s / self.app.env.n_cols)
                c = s % self.app.env.n_cols
                self.app.draw_policy(r, c, p)
            time.sleep(sleep_time)
        
        def test_policy(self, step, pi, num_of_tests=20):
            gamma = 1 if self.gamma is None else self.gamma
            total_reward = 0
            death = 0
            for i in range(num_of_tests):
                terminal, reward = False, 0
                s = self.app.env.reset()
                frames = 0
                while not terminal and frames < 99:
                    frames += 1
                    s, r, terminal, _ = self.app.env.step(pi[s])
                    reward = r + gamma * reward
                    if terminal and r != 1:
                        death += 1
                total_reward += reward
            x = np.append(self.test_plt.get_xdata(), step)
            y = np.append(self.test_plt.get_ydata(), total_reward/num_of_tests)
            self.test_plt.set_xdata(x)
            self.test_plt.set_ydata(y)
            self.test_plt_ax.relim()
            self.test_plt_ax.autoscale_view(True,True,True)
            # self.test_plt_ax.clear()
            # self.test_plt, = self.test_plt_ax.plot(x, y)
            self.test_plt_ax.set_xlabel("Iteration")
            self.test_plt_ax.set_ylabel("Accumulated Reward")
            self.value_plt.fig.canvas.draw()
            self.value_plt.fig.canvas.flush_events()
        
        def clear(self):
            self.test_plt_ax.clear()
            self.test_plt, = self.test_plt_ax.plot([], [])
            self.value_plt.clear()
            self.app.world_canvas.delete("policy")


    DEFAULT_WORLD = lambda : [
            ["_", "_", "_",  1],
            ["_", "o", "_", -1],
            ["s", "_", "_", "_"]
        ]

    def __init__(self, alg_fn_map, worlds, master=None):
        super().__init__(master)
        self.alg_fn_map = alg_fn_map
        self.worlds = worlds

        self.master.title("Markov Decision Process -- CPSC 4820/6820 Clemson University")
        self.master.geometry("800x800")
        self.master.resizable(False, False)

        self.logger = self.Logger(self)
        plot_toolbar_frame = tk.Frame(master=self.master)
        plot_canvas, _ = build_matplotlib_canvas(
            self.logger.value_plt.fig, self.master, plot_toolbar_frame
        )
        self.plot_canvas = plot_canvas.get_tk_widget()
        self.logger.value_plt.fig.set_canvas(plot_canvas)
        self.logger.init()

        self.world_canvas = tk.Canvas(self.master, bg="white", bd=0, highlightthickness=0, relief='ridge')
        self.text_alg = tk.Label(self.master, text=" ")
        self.text_gamma = tk.Label(self.master, text=" ")
        self.text_noise = tk.Label(self.master, text=" ")
     
        self.plot_canvas.grid(row=0, column=0, rowspan=4,
            sticky=tk.W+tk.N+tk.S, padx=10
        )
        plot_toolbar_frame.grid(row=5, column=0, 
            sticky=tk.W+tk.E, padx=10, pady=(10, 0)
        )
        self.world_canvas.grid(row=0, column=1,
            sticky=tk.W+tk.E+tk.N+tk.S, padx=10
        )
        self.text_alg.grid(row=1, column=1,
            sticky=tk.E, padx=10
        )
        self.text_gamma.grid(row=2, column=1,
            sticky=tk.E, padx=10
        )
        self.text_noise.grid(row=3, column=1,
            sticky=tk.E, padx=10
        )
    
        btn_frame = tk.Frame(master=self.master)
        btn_frame.grid(row=5, column=1,
            sticky=tk.E, padx=10, pady=(10, 10)
        )

        self.btn_solve = tk.Button(btn_frame, text="Solve", command=self.solve)
        self.btn_test = tk.Button(btn_frame, text="Test")
        self.btn_test.config(state=tk.DISABLED)
        self.btn_test.pack(side=tk.RIGHT, padx=10)
        self.btn_solve.pack(side=tk.RIGHT)

        self.master.grid_columnconfigure(0, weight=3)
        self.master.grid_columnconfigure(1, weight=2)
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=0)
        self.master.grid_rowconfigure(2, weight=0)
        self.master.grid_rowconfigure(3, weight=0)
        self.master.grid_rowconfigure(4, weight=0)
        self.master.grid_rowconfigure(5, weight=0)

        self.solve_window = None
        self.agent_icon = None
        self.cell_size = 0
        self.world_canvas.bind("<Configure>", lambda _: self.generate_world(next(v for k, v in self.worlds.items()), noise=0.2))

    def solve(self):
        if self.solve_window is not None:
            self.solve_window.update()
            self.solve_window.deiconify()
            return
        
        self.solve_window = tk.Toplevel(self.master)
        self.solve_window.title("Solver Options")
        self.solve_window.protocol("WM_DELETE_WINDOW", self.solve_window.withdraw)
        self.solve_window.resizable(False, False)

        alg_var = tk.StringVar(self.solve_window)
        alg_var.set(next(iter(self.alg_fn_map.keys())))
        listbox_alg = tk.OptionMenu(self.solve_window, alg_var, *self.alg_fn_map.keys())

        def int_validate(text, value_if_allowed):
            if len(value_if_allowed) > 0:
                try:
                    int(value_if_allowed)
                except ValueError:
                    return False
            return True
        max_iter_var = tk.StringVar(self.solve_window)
        max_iter_var.set("2000")
        field_max_iter = tk.Entry(self.solve_window,
            textvariable=max_iter_var,
            validate="key",
            validatecommand=(self.solve_window.register(int_validate), '%S', '%P')
        )

        gamma_var = tk.StringVar(self.solve_window)
        gamma_var.set("0.95")
        field_gamma = tk.Entry(self.solve_window, textvariable=gamma_var)
        
        noise_var = tk.StringVar(self.solve_window)
        noise_var.set("0.2")
        field_noise = tk.Entry(self.solve_window, textvariable=noise_var)

        world_var = tk.StringVar(self.solve_window)
        world_var.set(next(iter(self.worlds.keys())))
        listbox_world = tk.OptionMenu(self.solve_window, world_var, *self.worlds.keys())

        def stop_request():
            self.stop_request = True

        def test(pi):
            self.btn_solve.config(state=tk.DISABLED)
            self.btn_test.config(text="Stop")
            self.btn_test.config(state=tk.NORMAL)
            self.stop_request = False
            self.btn_test.config(command=stop_request)
            def test_thread():
                self.test_policy(pi)
                self.btn_solve.config(state=tk.NORMAL)
                self.btn_test.config(text="Test")
                self.btn_test.config(state=tk.NORMAL)
                self.btn_test.config(command=lambda: test(pi))
            Thread(target=test_thread).start()

        def solve():
            failed = False
            try:
                max_iter = int(max_iter_var.get())
            except ValueError:
                failed = True
            if failed or max_iter <= 0:
                tk.messagebox.showinfo("", "Max iterations must be an integer greater than 0.")
                return
            try:
                gamma = float(gamma_var.get())
            except ValueError:
                failed = True
            if failed or gamma < 0:
                tk.messagebox.showinfo("", "Gamma must be a non-negative number.")
                return
            try:
                noise = float(noise_var.get())
            except ValueError:
                failed = True
            if failed or noise < 0 or noise >= 1:
                tk.messagebox.showinfo("", "Noise must be a non-negative float less than 1.")
                return
       
            btn_submit.config(state=tk.DISABLED)
            self.btn_solve.config(state=tk.DISABLED)
            self.btn_test.config(state=tk.DISABLED)
            self.btn_test.config(command=lambda: None)
            self.solve_window.withdraw()
            
            self.logger.clear()
            self.logger.gamma = gamma
            solver = self.alg_fn_map[alg_var.get()]
            world = self.worlds[world_var.get()]

            self.generate_world(world, noise)
            self.text_alg.config(text=alg_var.get())
            self.text_gamma.config(text="Reward Discount Factor: {}".format(gamma))
            self.text_noise.config(text="Noise: {}".format(noise))
       
            pi = solver(self.env, gamma=gamma, max_iterations=max_iter, logger=self.logger)

            if pi is None:
                self.btn_solve.config(state=tk.NORMAL)
                tk.messagebox.showinfo("", "No policy received.")
            else:
                valid = True
                for s in range(self.env.observation_space.n):
                    if s not in pi or pi[s] not in range(self.env.action_space.n):
                        valid = False
                    if not valid:
                        break
                if valid:
                    self.btn_solve.config(state=tk.NORMAL)
                    tk.messagebox.showinfo("", "Invalid policy received.")
                else:
                    self.logger.draw_policy(pi)
                    btn_submit.config(state=tk.NORMAL)
                    self.btn_solve.config(state=tk.NORMAL)
                    self.btn_test.config(state=tk.NORMAL)
                    self.btn_test.config(command=lambda: test(pi))
                    tk.messagebox.showinfo("", "Optimization is done.")

        btn_submit = tk.Button(self.solve_window, text="Done", command=solve)

        tk.Label(self.solve_window, text="Algorithm").grid(
            row=0, column=0, padx=10, sticky=tk.E
        )
        listbox_alg.grid(
            row=0, column=1, padx=10, sticky=tk.W
        )
        tk.Label(self.solve_window, text="Max Iterations").grid(
            row=1, column=0, padx=10, sticky=tk.E
        )
        field_max_iter.grid(
            row=1, column=1, padx=10, sticky=tk.W
        )
        tk.Label(self.solve_window,
            text="Reward Discount Factor\n(gamma)", justify=tk.RIGHT).grid(
            row=2, column=0, padx=10, sticky=tk.E
        )
        field_gamma.grid(
            row=2, column=1, padx=10, sticky=tk.W
        )
        tk.Label(self.solve_window,
            text="Action Noise", justify=tk.RIGHT).grid(
            row=3, column=0, padx=10, sticky=tk.E
        )
        field_noise.grid(
            row=3, column=1, padx=10, sticky=tk.W
        )
        tk.Label(self.solve_window,
            text="World", justify=tk.RIGHT).grid(
            row=5, column=0, padx=10, sticky=tk.E
        )
        listbox_world.grid(
            row=5, column=1, padx=10, sticky=tk.W
        )

        btn_submit.grid(
            row=6, columnspan=2, pady=(20, 0)
        )
    

    def draw_world(self):
        self.world_canvas.delete("all")

        w = self.world_canvas.winfo_width()-2
        h = self.world_canvas.winfo_height()-2
        cell_size = min(w/self.env.n_cols, h/self.env.n_rows)
        if cell_size != self.cell_size:
            self.agent_icon = None
            self.cell_size = cell_size
        self.board_pos = (
            (w - self.cell_size*self.env.n_cols)*0.5,
            (h - self.cell_size*self.env.n_rows)*0.5,
            (w + self.cell_size*self.env.n_cols)*0.5,
            (h + self.cell_size*self.env.n_rows)*0.5,
        )

        for r in range(self.env.n_rows+1):
            self.world_canvas.create_line(
                self.board_pos[0], self.board_pos[1] + r*self.cell_size,
                self.board_pos[2], self.board_pos[1] + r*self.cell_size,
            )
        for c in range(self.env.n_cols+1):
            self.world_canvas.create_line(
                self.board_pos[0] + c*self.cell_size, self.board_pos[1],
                self.board_pos[0] + c*self.cell_size, self.board_pos[3],
            )
        
        for r, row in enumerate(self.env.reward_map):
            for c, d in enumerate(row):
                s = r*self.env.n_cols + c
                if s in self.env.starts:
                    self.draw_start(r, c)
                elif s in self.env.obstacles:
                    self.draw_obstacle(r, c)
                elif s in self.env.goals:
                    if d > 0:
                        self.draw_goal(r, c)
                    else:
                        self.draw_hole(r, c)
        
    def draw_start(self, r, c):
        self.world_canvas.create_rectangle(
            self.board_pos[0]+self.cell_size*c, self.board_pos[1]+self.cell_size*r,
            self.board_pos[0]+self.cell_size*(c+1), self.board_pos[1]+self.cell_size*(r+1),
            fill="deep sky blue", tag="start"
        )
        self.world_canvas.create_text(
            self.board_pos[0]+self.cell_size*(c+0.5), self.board_pos[1]+self.cell_size*(r+0.5),
            text="S", fill="white",
            font=(None, int(math.ceil(min(self.cell_size, self.cell_size)*0.5))),
            tag="start"
        )
    
    def draw_goal(self, r, c):
        self.world_canvas.create_rectangle(
            self.board_pos[0]+self.cell_size*c, self.board_pos[1]+self.cell_size*r,
            self.board_pos[0]+self.cell_size*(c+1), self.board_pos[1]+self.cell_size*(r+1),
            fill="lime green", tag="goal"
        )
        text = "+" + str(self.env.reward_map[r][c])
        self.world_canvas.create_text(
            self.board_pos[0]+self.cell_size*(c+0.5), self.board_pos[1]+self.cell_size*(r+0.5),
            text=text, fill="white",
            font=(None, int(math.ceil(min(self.cell_size*0.98/len(text), self.cell_size*0.2)))),
            tag="goal"
        )

    def draw_obstacle(self, r, c):
        self.world_canvas.create_rectangle(
            self.board_pos[0]+self.cell_size*c, self.board_pos[1]+self.cell_size*r,
            self.board_pos[0]+self.cell_size*(c+1), self.board_pos[1]+self.cell_size*(r+1),
            fill="gray", tag="obstacle"
        )
    
    def draw_hole(self, r, c):
        self.world_canvas.create_rectangle(
            self.board_pos[0]+self.cell_size*c, self.board_pos[1]+self.cell_size*r,
            self.board_pos[0]+self.cell_size*(c+1), self.board_pos[1]+self.cell_size*(r+1),
            fill="red", tag="hole"
        )
        if self.env.reward_map[r][c] != 0:
            text = str(self.env.reward_map[r][c])
            self.world_canvas.create_text(
                self.board_pos[0]+self.cell_size*(c+0.5), self.board_pos[1]+self.cell_size*(r+0.5),
                text=text, fill="white",
                font=(None, int(math.ceil(min(self.cell_size*0.98/len(text), self.cell_size*0.2)))),
                tag="hole"
            )
    
    def draw_policy(self, r, c, a):
        r += 0.5
        c += 0.5
        if a == 0:
            r_, c_ = r-0.5, c
        elif a == 1:
            r_, c_ = r, c+0.5
        elif a == 2:
            r_, c_ = r+0.5, c
        else:
            r_, c_ = r, c-0.5

        self.world_canvas.create_line(
            self.board_pos[0]+self.cell_size*c,  self.board_pos[1]+self.cell_size*r,
            self.board_pos[0]+self.cell_size*c_, self.board_pos[1]+self.cell_size*r_,
            arrow=tk.LAST, tag="policy"
        )
    
    def draw_agent(self, r, c):
        if self.agent_icon is None:
            self.agent_icon = tk.PhotoImage(file=os.path.join(os.path.dirname(os.path.realpath(__file__)), "clemson.png"))
            sx = int(self.agent_icon.width()/(self.cell_size*0.8))
            sy = int(self.agent_icon.height()/(self.cell_size*0.8))
            self.agent_icon = self.agent_icon.subsample(sx, sy)
        self.world_canvas.delete("agent")
        self.world_canvas.create_image(
            self.board_pos[0]+self.cell_size*(c+0.5),
            self.board_pos[1]+self.cell_size*(r+0.5),
            anchor=tk.CENTER, image=self.agent_icon,
            tag="agent"
        )

    def test_policy(self, policy):
        def redraw(s):
            r = int(s/self.env.n_cols)
            c = s%self.env.n_cols
            self.draw_agent(r, c)
            self.world_canvas.update()
            # self.env.render()
        terminal = False
        frames = 0
        s = self.env.reset()
        redraw(s)
        self.world_canvas.update()
        while not terminal and frames < 1000 and not self.stop_request:
            frames += 1
            time.sleep(0.15)
            s, r, terminal, _ = self.env.step(policy[s])
            redraw(s)
        if self.stop_request:
            print("Test terminates.")
        elif terminal:
            print("Finished in {} steps.".format(frames))
        else:
            print("Overtime")
            

    def generate_world(self, generate_fn, noise):
        world_map = generate_fn()
        goal = []
        start = []
        obstacles = []
        reward_map = []
        for r, row in enumerate(world_map):
            rew_ = []
            for c, d in enumerate(row):
                s = c+r*len(row)
                if d == 's':
                    start.append(s)
                if isinstance(d, int) or isinstance(d, float):
                    goal.append(s)
                    rew = d
                elif d == 'o':
                    obstacles.append(s)
                    rew = 0
                else:
                    rew = 0
                rew_.append(rew)
            reward_map.append(rew_)

        self.env = GridWorld(reward_map, starts=start, goals=goal, obstacles=obstacles, noise=noise)
        self.logger.clear()
        self.draw_world()
