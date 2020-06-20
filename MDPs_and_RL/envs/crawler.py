# crawler.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) 
import math
import numpy as np

class CrawlerEnv(object):

    def __init__(self, render=False):
        if render:
            raise NotImplementedError
        self.base_width = 80
        self.base_height = 40
        self.upper_arm_length = 60
        self.forearm_length = 40
        self.links = {
            "base": {
                "world_position": (self.base_width*0.5, self.base_height*0.5),
                "world_rotation": 0,
                "shape": (
                    (-self.base_width*0.5, self.base_height*0.5), (self.base_width*0.5, self.base_height*0.5),
                    (self.base_width*0.5, -self.base_height*0.5), (-self.base_width*0.5, -self.base_height*0.5)
                )
            },
            "upper_arm": {
                "parent": "base",
                "joint_position_on_parent": (self.base_width*0.5, self.base_height*0.5), 
                "joint_rotation_on_parent": 0,
                "position": (self.upper_arm_length*0.5, 0),
                "rotation": 0,
                "shape": (
                    (-self.upper_arm_length*0.5, 5), (self.upper_arm_length*0.5, 5),
                    (self.upper_arm_length*0.5, -5), (-self.upper_arm_length*0.5, -5)
                )
            },
            "forearm": {
                "parent": "upper_arm",
                "joint_position_on_parent": (self.upper_arm_length*0.5, 0), 
                "joint_rotation_on_parent": 0,
                "position": (self.forearm_length*0.5, 0),
                "rotation": 0,
                "shape": (
                    (-self.forearm_length*0.5, 3), (self.forearm_length*0.5, 0),
                    (-self.forearm_length*0.5, -3)
                )
            },
            "hand": {
                "parent": "forearm",
                "joint_position_on_parent": (self.forearm_length*0.5, 0), 
                "joint_rotation_on_parent": 0,
                "position": (0, 0),
                "rotation": 0,
                "shape": ()
            }
        }
        self.joints = {
            "upper_arm": {
                "position": 0,
                "limitation": (-math.pi/6, math.pi/6),
                "space": 7
            },
            "forearm": {
                "position": 0,
                "limitation": (-5*math.pi/6, 0),
                "space": 11
            },
            "hand": {
                "position": 0,
                "limitation": (0, 0),
                "space": 0
            }
        }
        for k in self.joints.keys():
            lower, upper = self.joints[k]["limitation"]
            self.joints[k]["space_size"] = (upper-lower)/max(1, self.joints[k]["space"])
        self.reset()

        class Discrete(object):
            def __init__(self, n):
                self.n = n
            def sample(self):
                return np.random.randint(self.n)
                
        self.action_space = Discrete(4)
        self.observation_space = Discrete((self.joints["upper_arm"]["space"]+1)*(self.joints["forearm"]["space"]+1))

        self.trans_model = [
            [None for a in range(self.action_space.n)]
            for s in range(self.observation_space.n)
        ]
        for s in range(self.observation_space.n):
            self.compute_foreward_kinematics(s)
            hand_pos = self.links["hand"]["world_position"]
            for a in range(self.action_space.n):
                s_ = self.new_state(s, a)
                if s == s_:
                    d = 0
                else:
                    self.compute_foreward_kinematics(s_)
                    hand_pos_ = self.links["hand"]["world_position"]
                    d = self.compute_movement(hand_pos, hand_pos_)
                self.trans_model[s][a] = [
                    # p(s'|s,a), s', R(s,a,s'), terminal 
                    (1.0, s_, d, False)
                ]

    def reset(self):
        self.state = 0
        self.links["base"]["world_position"] = (self.base_width*0.5, self.base_height*0.5)
        self.links["base"]["world_rotation"] = 0
        self.compute_foreward_kinematics(self.state)
        self.hand_pos = self.links["hand"]["world_position"]
        self.last_movement = 0
        self.steps = 0
        return self.state

    def step(self, a):
        assert(a in range(4))
        self.steps += 1 
        horizontal_pos = self.links["base"]["world_position"][0]
        hand_pos = self.hand_pos

        s_ = self.new_state(self.state, a)
        if s_ == self.state:
            self.last_movement = 0
        else:
            self.state = s_
            # fix the base and compute the forward kinematics
            # then, we decide the base rotation and movement by assuming that 
            # the left-bottom corner of the base is always on the ground and the hand is always above or on the ground
            self.links["base"]["world_position"] = (self.base_width*0.5, self.base_height*0.5)
            self.links["base"]["world_rotation"] = 0
            self.compute_foreward_kinematics(self.state)
            self.hand_pos = self.links["hand"]["world_position"]
            # compute movement
            self.last_movement = self.compute_movement(hand_pos, self.hand_pos)
            # compute base rotation
            if self.links["hand"]["world_position"][1] < 0:
                self.links["base"]["world_rotation"] = math.atan(-self.hand_pos[1]/self.hand_pos[0])
            else:
                self.links["base"]["world_rotation"] = 0
            # update the position and rotation of all links
            self.links["base"]["world_position"] = (
                horizontal_pos+self.last_movement,
                self.rotate(self.links["base"]["world_position"], self.links["base"]["world_rotation"])[1]
            )
            delta_pos = np.subtract(self.links["base"]["world_position"], (self.base_width*0.5, self.base_height*0.5))
            for l in ["upper_arm", "forearm", "hand"]:
                self.links[l]["joint_world_rotation"] += self.links["base"]["world_rotation"]
                self.links[l]["world_rotation"] += self.links["base"]["world_rotation"]
                self.links[l]["joint_world_position"] = np.add(
                    delta_pos,
                    self.rotate(self.links[l]["joint_world_position"], self.links["base"]["world_rotation"])
                )
                self.links[l]["world_position"] = np.add(
                    delta_pos,
                    self.rotate(self.links[l]["world_position"], self.links["base"]["world_rotation"])
                )
        return self.state, self.last_movement, False, {}

    def render(self, canvas=None):
        if canvas is None:
            if hasattr(self, "canvas"):
                canvas = self.canvas
            else:
                raise NotImplementedError
        ground_thinkness = 20
        ground_line_gap = 30
        if not hasattr(self, "ground_pos"):
            self.ground_pos = 0

        w = canvas.winfo_width()
        h = canvas.winfo_height()-ground_thinkness
        if self.links["base"]["world_position"][0] < self.base_width*0.5:
            d = self.base_width*0.5-self.links["base"]["world_position"][0]
            self.ground_pos += d
        elif self.links["base"]["world_position"][0] > w-(self.upper_arm_length+self.forearm_length+self.base_width*0.5):
            d = w-(self.upper_arm_length+self.forearm_length+self.base_width*0.5)-self.links["base"]["world_position"][0]
            self.ground_pos += d
        else:
            d = 0
        if d != 0:
            for l in self.links.keys():
                self.links[l]["world_position"] = (self.links[l]["world_position"][0] + d, self.links[l]["world_position"][1])
                if "joint_world_position" in self.links[l]:
                    self.links[l]["joint_world_position"] = (self.links[l]["joint_world_position"][0] + d, self.links[l]["joint_world_position"][1])

        self.ground_pos = self.ground_pos % ground_line_gap

        # draw base
        vertex = []
        for p in self.links["base"]["shape"]:
            p_ = np.add(
                self.links["base"]["world_position"],
                self.rotate(p, self.links["base"]["world_rotation"])
            )
            vertex.append(p_[0])
            vertex.append(h-p_[1])
        canvas.create_polygon(*vertex,
            fill="saddle brown"
        )
        # draw upper arm
        vertex = []
        for p in self.links["upper_arm"]["shape"]:
            p_ = np.add(
                self.links["upper_arm"]["world_position"],
                self.rotate(p, self.links["upper_arm"]["world_rotation"])
            )
            vertex.append(p_[0])
            vertex.append(h-p_[1])
        canvas.create_polygon(*vertex,
            fill="sandy brown"
        )
        x, y = self.links["upper_arm"]["joint_world_position"]
        canvas.create_oval(
            x-1, h-y-1,
            x+1, h-y+1,
            fill="black"
        )
        # draw forearm
        vertex = []
        for p in self.links["forearm"]["shape"]:
            p_ = np.add(
                self.links["forearm"]["world_position"],
                self.rotate(p, self.links["forearm"]["world_rotation"])
            )
            vertex.append(p_[0])
            vertex.append(h-p_[1])
        canvas.create_polygon(*vertex,
            fill="sandy brown"
        )
        x, y = self.links["forearm"]["joint_world_position"]
        canvas.create_oval(
            x-1, h-y-1,
            x+1, h-y+1,
            fill="black"
        )
        # draw ground
        canvas.create_line(
            0, h, w, h, fill="black"
        )
        for x in range(0, w+ground_line_gap, ground_line_gap):
            canvas.create_line(
                self.ground_pos+x, h, self.ground_pos+x-ground_line_gap, h+ground_thinkness, fill="black"
            )
        # text info
        canvas.create_text(100, 10,
            text="Steps: {}".format(self.steps)
        )

    def compute_foreward_kinematics(self, s):
        s0 = int(s/(self.joints["forearm"]["space"]+1))
        s1 = s%(self.joints["forearm"]["space"]+1)
        self.joints["upper_arm"]["position"] = s0*self.joints["upper_arm"]["space_size"] + self.joints["upper_arm"]["limitation"][0]
        self.joints["forearm"]["position"] = s1*self.joints["forearm"]["space_size"] + self.joints["forearm"]["limitation"][0]
        for l in ["upper_arm", "forearm", "hand"]:
            p = self.links[self.links[l]["parent"]]
            self.links[l]["joint_world_rotation"] = p["world_rotation"] + self.links[l]["joint_rotation_on_parent"]
            self.links[l]["joint_world_position"] = np.add(
                p["world_position"],
                self.rotate(self.links[l]["joint_position_on_parent"], p["world_rotation"])
            )
            self.links[l]["world_rotation"] = self.links[l]["joint_world_rotation"] + self.joints[l]["position"] + self.links[l]["rotation"]
            self.links[l]["world_position"] = np.add(
                self.links[l]["joint_world_position"],
                self.rotate(self.links[l]["position"], self.links[l]["world_rotation"])
            )

    @staticmethod
    def compute_movement(hand_pos_old, hand_pos_new):
        if hand_pos_old[1] <= 0 and hand_pos_new[1] < 0:
            d = np.linalg.norm(hand_pos_old) - np.linalg.norm(hand_pos_new)
        elif hand_pos_old[1] >= 0 and hand_pos_new[1] >= 0:
            d = 0
        elif hand_pos_old[1] > 0 and hand_pos_new[1] < 0:
            # only count the movement for the part when hand_pos_old[1] < 0
            d = (hand_pos_old[0] - hand_pos_old[1]*(hand_pos_new[0]-hand_pos_old[0])/(hand_pos_new[1]-hand_pos_old[1])) - np.linalg.norm(hand_pos_new)
        else: #if hand_pos_old[1] < 0 and hand_pos_new[1] >= 0:
            d = np.linalg.norm(hand_pos_old) - (hand_pos_new[0] - hand_pos_new[1]*(hand_pos_old[0]-hand_pos_new[0])/(hand_pos_old[1]-hand_pos_new[1]))
        return d

    def new_state(self, s, a):
        s0 = int(s/(self.joints["forearm"]["space"]+1))
        s1 = s%(self.joints["forearm"]["space"]+1)
        s0 = max(0, min(self.joints["upper_arm"]["space"], s0 + int((a+1)%2)*(a-1)))
        s1 = max(0, min(self.joints["forearm"]["space"], s1 + int(a%2)*(a-2)))
        s_ = s0*(self.joints["forearm"]["space"]+1) + s1
        return s_

    @staticmethod
    def rotate(p, angle, around=(0,0)):
        if angle == 0:
            return p
        c, s = math.cos(angle), math.sin(angle)
        rot = [
            [c, -s],
            [s, c]
        ]
        return np.add(
            around,
            np.matmul(rot, np.subtract(p, around))
        )    
    
