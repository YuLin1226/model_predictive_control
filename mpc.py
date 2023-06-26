#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import pathlib


class ModelPredictiveControl:

    def __init__(self) -> None:
        
        self.HL_ = 5 # Horizon length
        self.NX_ = 3 # Number of design variables: x, y, yaw
        self.NU_ = 4 # Number of control inputs: front steer & speed, rear steer & speed

        self.R_ = np.diag([0.01, 0.01, 0.01, 0.01])
        self.Rd_ = np.diag([0.001, 0.001, 0.001, 0.001])
        self.Q_ = np.diag([0.01, 0.01, 0.01, 0.01])
        self.Qf_ = np.diag([0.001, 0.001, 0.001, 0.001])

        self.MAX_TRAVEL_SPEED_ = 0.5
        self.MAX_STEER_SPEED_ = 0.5
        self.DT_ = 0.1

        self.isModelParameterRetrived_ = False
        self.wheel_base_ = 0.0

    def start(self):
        
        if not self.isModelParameterRetrived_:
            print("Error: Model Parameters haven't been retrived.")
            return False

    def controlLaw(self, x_ref, x_init):

        x = cvxpy.Variable((self.NX_, self.HL_ + 1))
        u = cvxpy.Variable((self.NU_, self.HL_))

        cost = 0.0
        constraints = []

        for t in range(self.HL_):
            cost += cvxpy.quad_form(u[:, t], self.R_)

            if t != 0:
                cost += cvxpy.quad_form(x_ref[:, t] - x[:, t], self.Q_)


            V = self.getVelocitiesFromRobotModels()
            constraints += [x[0, t + 1] == x[0, t] + V[0] * self.DT_]
            constraints += [x[1, t + 1] == x[1, t] + V[1] * self.DT_]
            constraints += [x[2, t + 1] == x[2, t] + V[2] * self.DT_]

            if t < (self.HL_ - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd_)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= self.MAX_STEER_SPEED_ * self.DT_]
                constraints += [cvxpy.abs(u[3, t + 1] - u[3, t]) <= self.MAX_STEER_SPEED_ * self.DT_]

        cost += cvxpy.quad_form(x_ref[:, self.HL_] - x[:, self.HL_], self.Qf_)

        constraints += [x[:, 0] == x_init]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_TRAVEL_SPEED_]
        constraints += [cvxpy.abs(u[2, :]) <= self.MAX_TRAVEL_SPEED_]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            optimized_x = np.array(x.value[0, :]).flatten()
            optimized_y = np.array(x.value[1, :]).flatten()
            optimized_v = np.array(x.value[2, :]).flatten()
            optimized_yaw = np.array(x.value[3, :]).flatten()
            optimized_a = np.array(u.value[0, :]).flatten()
            optimized_delta = np.array(u.value[0, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            optimized_x, optimized_y, optimized_v, optimized_yaw, optimized_a, optimized_delta = None, None, None, None, None, None

        return optimized_x, optimized_y, optimized_v, optimized_yaw, optimized_a, optimized_delta
        

    def predictMotion(self):
        pass

    def getVelocitiesFromRobotModels(self, front_steer, front_speed, rear_steer, rear_speed):

        if not self.isModelParameterRetrived_:
            print("Error: Model Parameters haven't been retrived.")
            return None
        
        H = np.array([
            [1, 0, 0]
            [0, 1, self.wheel_base_/2],
            [1, 0, 0],
            [0, 1, -self.wheel_base_/2]
        ])
        v1x = math.cos(front_steer) * front_speed
        v1y = math.sin(front_steer) * front_speed
        v2x = math.cos(rear_steer) * rear_speed
        v2y = math.sin(rear_steer) * rear_speed
        vo = np.array([
            [v1x],
            [v1y],
            [v2x],
            [v2y]
        ])
        vi = (H.transpose() @ H).inverse() @ H.transpose() @ vo
        return vi


    def setRobotModelParameters(self, wheel_base):

        self.isModelParameterRetrived_ = True
        self.wheel_base_ = wheel_base

