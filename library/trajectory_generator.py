#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import csv
import numpy as np
import math
from .polynomial import Polynomial
from .general_bicycle_model import GeneralBicycleModel


class TrajectoryGenerator:

    def __init__(self) -> None:
        self.gbm_ = GeneralBicycleModel(wheel_base=1)

    def retriveTrajectoryFromCSV(self, file_name):
        node_lists = []
        with open(file_name, newline='') as csvfile:
            rows = csv.reader(csvfile)
            skip_first = True
            for row in rows:
                if skip_first:
                    skip_first = False
                    continue
                node = []
                for i, element in enumerate(row):    
                    if i == 4:
                        # node[4] is mode type, which is a string.
                        node.append(element)
                    else:
                        node.append(float(element))
                node_lists.append(node)
        return node_lists

    def interpolateTrajectory(self, node_lists, num_interval=100):
        pts_x, pts_y, pts_yaw, pts_mode = [], [], [], []
        pts_vf, pts_vr, pts_sf, pts_sr = [], [], [], []
        # each element in node_lists has 2 polynomial sets.
        for i, node in enumerate(node_lists):
            if i == 0:
                pts_x.append(node[0])
                pts_y.append(node[1])
                pts_yaw.append(node[2])
                pts_mode.append(node[4])
                continue
            front_wheel_travel_poly_1 = Polynomial(node[5], node[6], node[7])
            front_wheel_travel_poly_2 = Polynomial(node[8], node[9], node[10])
            front_wheel_steer_poly_1 = Polynomial(node[11], node[12], node[13])
            front_wheel_steer_poly_2 = Polynomial(node[14], node[15], node[16])
            rear_wheel_travel_poly_1 = Polynomial(node[17], node[18], node[19])
            rear_wheel_travel_poly_2 = Polynomial(node[20], node[21], node[22])
            rear_wheel_steer_poly_1 = Polynomial(node[23], node[24], node[25])
            rear_wheel_steer_poly_2 = Polynomial(node[26], node[27], node[28])
            T = node[3] - node_lists[i-1][3]
            half_T = T / 2
            ts = np.linspace(0, half_T, num_interval) 
            dt = np.average(np.diff(ts))
            x, y, yaw = node_lists[i-1][0], node_lists[i-1][1], node_lists[i-1][2]
            for t in ts:
                x, y, yaw, vf, sf, vr, sr = self.computeNextNodePose(x, y, yaw, front_wheel_travel_poly_1, rear_wheel_travel_poly_1, front_wheel_steer_poly_1, rear_wheel_steer_poly_1, t, dt)
                pts_x.append(x)
                pts_y.append(y)
                pts_yaw.append(yaw)
                pts_mode.append(node[4])
                pts_vf.append(vf)
                pts_sf.append(sf)
                pts_vr.append(vr)
                pts_sr.append(sr)
            for t in ts:
                x, y, yaw, vf, sf, vr, sr = self.computeNextNodePose(x, y, yaw, front_wheel_travel_poly_2, rear_wheel_travel_poly_2, front_wheel_steer_poly_2, rear_wheel_steer_poly_2, t, dt)
                pts_x.append(x)
                pts_y.append(y)
                pts_yaw.append(yaw)
                pts_mode.append(node[4])
                pts_vf.append(vf)
                pts_sf.append(sf)
                pts_vr.append(vr)
                pts_sr.append(sr)
        return pts_x, pts_y, pts_yaw, pts_mode, pts_vf, pts_vr, pts_sf, pts_sr

    def computeNextNodePose(self, last_x:float, last_y:float, last_yaw:float, front_wheel_travel_poly:Polynomial, rear_wheel_travel_poly:Polynomial, front_wheel_steer_poly:Polynomial, rear_wheel_steer_poly:Polynomial, t:float, dt:float):
        
        front_speed = front_wheel_travel_poly.a_ * t**2 + front_wheel_travel_poly.b_ * t + front_wheel_travel_poly.c_
        front_steer = front_wheel_steer_poly.a_ * t**2 + front_wheel_steer_poly.b_ * t + front_wheel_steer_poly.c_
        rear_speed = rear_wheel_travel_poly.a_ * t**2 + rear_wheel_travel_poly.b_ * t + rear_wheel_travel_poly.c_
        rear_steer = rear_wheel_steer_poly.a_ * t**2 + rear_wheel_steer_poly.b_ * t + rear_wheel_steer_poly.c_
        vx, vy, w = self.gbm_.transformWheelCommandToRobotCommand(
            vf=front_speed,
            sf=front_steer,
            vr=rear_speed,
            sr=rear_steer
        )
        x = last_x + (vx * math.cos(last_yaw) - vy * math.sin(last_yaw)) * dt
        y = last_y + (vx * math.sin(last_yaw) + vy * math.cos(last_yaw)) * dt
        yaw = last_yaw + w * dt
        return x, y, yaw, front_speed, front_steer, rear_speed, rear_steer

    def splitTrajectoryWithMotionModes(self, cmode):

        trajectories_idx_group = []
        from_idx, to_idx = None, None
        current_mode = None
        for i, mode in zip(range(len(cmode) - 1), cmode):
            
            if from_idx is None:
                from_idx = i
                current_mode = mode

            if current_mode != cmode[i+1]:
                to_idx = i
                trajectories_idx_group.append([from_idx, to_idx])
                from_idx, to_idx = None, None

        return trajectories_idx_group

    def compressTrajectory(self, pts_x, pts_y, pts_yaw, pts_mode, pts_vf, pts_vr, pts_sf, pts_sr, step=5):
        pts_x = pts_x[::step]
        pts_y = pts_y[::step]
        pts_yaw = pts_yaw[::step]
        pts_mode = pts_mode[::step]
        pts_vf = pts_vf[::step]
        pts_vr = pts_vr[::step]
        pts_sf = pts_sf[::step]
        pts_sr = pts_sr[::step]
        return pts_x, pts_y, pts_yaw, pts_mode, pts_vf, pts_vr, pts_sf, pts_sr

def test1():
    tg = TrajectoryGenerator()
    node_lists = tg.retriveTrajectoryFromCSV("g2_cm_path.csv")
    X, Y, YAW, MODE, VF, SF, VR, SR = tg.interpolateTrajectory(node_lists)

if __name__ == '__main__':
    test1()