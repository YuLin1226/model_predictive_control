#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import csv
import sys
import pathlib


class State:

    def __init__(self, x, y, yaw, vx, vy, w, front_steer, front_speed, rear_steer, rear_speed) -> None:
        
        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw
        self.vx_ = vx
        self.vy_ = vy
        self.w_ = w
        self.front_steer_ = front_steer
        self.front_speed_ = front_speed
        self.rear_steer_ = rear_steer
        self.rear_speed_ = rear_speed


class ModelPredictiveControl:

    def __init__(self) -> None:
        
        self.ITERATION_TIMES_ = 1
        self.HL_ = 5 # Horizon length
        self.NX_ = 3 # Number of design variables: x, y, yaw
        self.NU_ = 4 # Number of control inputs: front steer & speed, rear steer & speed

        self.R_ = np.diag([0.01, 0.01, 0.01, 0.01])
        self.R_diff_ = np.diag([0.01, 0.01, 0.01, 0.01])
        self.Q_ = np.diag([0.01, 0.01, 0.01])
        self.Q_final_ = np.diag([0.01, 0.01, 0.01])

        self.LOOKAHEAD_DIST_ = 1
        self.MAX_TRAVEL_SPEED_ = 0.5
        self.MAX_STEER_SPEED_ = 0.5
        self.DT_ = 0.1

        self.isModelParameterRetrived_ = False
        self.wheel_base_ = 0.0
        self.kinematical_matrix_ = None

        self.isReferenceRetrived_ = False
        self.reference_ = None
        self.trim_dist_ = 1

    def initialization(self, wheel_base, file_name):

        self.setRobotModelParameters(wheel_base=wheel_base)
        self.retriveReferenceFromCSV(file_name=file_name)

    def start(self, current_pos_x, current_pos_y, current_pos_yaw):
        
        if not self.isModelParameterRetrived_:
            print("Error: Model Parameters haven't been retrived.")
            return False
        
        if not self.isReferenceRetrived_:
            print("Error: Reference hasn't been retrived.")
            return False
            
        self.control_input_ = []
        if not self.control_input_:
            for i in range(self.HL_):
                self.control_input_.append(
                    [0, 0, 0, 0]
                )

        for i in range(self.ITERATION_TIMES_):
            x_current = self.updateCurrentState(x=current_pos_x, y=current_pos_y, yaw=current_pos_yaw)
            x_ref = self.getReferenceTrajectoryWithinHorizon(x_current=x_current)
            x_predicted = self.predictMotion(x_ref=x_ref, x_current=x_current, control_input=self.control_input_)

            opt_x, opt_y, opt_yaw, opt_front_speed, opt_front_steer, opt_rear_speed, opt_rear_steer = self.controlLaw(
                x_ref=x_ref,
                x_current=x_current,
                x_predicted=x_predicted,
                u_predicted=np.array(self.control_input_).transpose()
            )

            self.control_input_ = []
            for i in range(self.HL_):
                self.control_input_.append(
                    [opt_front_speed[i], opt_front_steer[i], opt_rear_speed[i], opt_rear_steer[i]]
                )

            output_v = self.getVelocitiesFromRobotModels(
                        front_speed=opt_front_speed[0],
                        front_steer=opt_front_steer[0],
                        rear_speed=opt_rear_speed[0],
                        rear_steer=opt_rear_steer[0]
                    )
        
        print("Optimized Front Speed : %f" %opt_front_speed[0])
        print("Optimized Front Steer : %f" %opt_front_steer[0])
        print("Optimized Rear Speed : %f" %opt_rear_speed[0])
        print("Optimized Rear Steer : %f" %opt_rear_steer[0])
        return output_v

    def controlLaw(self, x_ref, x_current, x_predicted, u_predicted):
        """
        x_ref       : array, size (NX, HL+1)
        x_current   : array, size (NX,    1)
        x_predicted : array, size (NX, HL+1)
        u_predicted : array, size (NU,   HL)  
        """
        x = cvxpy.Variable((self.NX_, self.HL_ + 1))
        u = cvxpy.Variable((int(self.NU_), self.HL_))

        cost = 0.0
        constraints = []

        for t in range(self.HL_):
            cost += cvxpy.quad_form(u[:, t], self.R_)

            if t != 0:
                cost += cvxpy.quad_form(x_ref[:, t] - x[:, t], self.Q_)

            A, B, C = self.getRobotModelMatrice(theta=x_predicted[2, t],
                                                v_f=u_predicted[0, t],
                                                delta_f=u_predicted[1, t],
                                                v_r=u_predicted[2, t],
                                                delta_r=u_predicted[3, t])
        
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            # if t == 0:
            #     constraints += [cvxpy.abs(u[1, t] - u_predicted[1, t]) <= self.MAX_STEER_SPEED_ * self.DT_]
            #     constraints += [cvxpy.abs(u[3, t] - u_predicted[3, t]) <= self.MAX_STEER_SPEED_ * self.DT_]
                
            if t < (self.HL_ - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.R_diff_)
                constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= self.MAX_STEER_SPEED_ * self.DT_]
                constraints += [cvxpy.abs(u[3, t+1] - u[3, t]) <= self.MAX_STEER_SPEED_ * self.DT_]
                
        cost += cvxpy.quad_form(x_ref[:, self.HL_] - x[:, self.HL_], self.Q_final_)

        x0 = [x_current[0,0], x_current[1,0], x_current[2,0]]

        constraints += [x[:, 0] == x0]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_TRAVEL_SPEED_]
        constraints += [cvxpy.abs(u[2, :]) <= self.MAX_TRAVEL_SPEED_]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            optimized_x   = np.array(x.value[0, :]).flatten() # this is only used in Plotting.
            optimized_y   = np.array(x.value[1, :]).flatten() # this is only used in Plotting.
            optimized_yaw = np.array(x.value[2, :]).flatten() # this is only used in Plotting.
            optimized_front_speed = np.array(u.value[0, :]).flatten()
            optimized_front_steer = np.array(u.value[1, :]).flatten()
            optimized_rear_speed  = np.array(u.value[2, :]).flatten()
            optimized_rear_steer  = np.array(u.value[3, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            optimized_x = None
            optimized_y = None
            optimized_yaw = None
            optimized_front_speed = None
            optimized_front_steer = None
            optimized_rear_speed = None
            optimized_rear_steer = None

        return optimized_x, optimized_y, optimized_yaw, optimized_front_speed, optimized_front_steer, optimized_rear_speed, optimized_rear_steer
        
    def getVelocitiesFromRobotModels(self, front_steer, front_speed, rear_steer, rear_speed):

        if not self.isModelParameterRetrived_:
            print("Error: Model Parameters haven't been retrived.")
            return None
        
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
        vi = self.kinematical_matrix_ @ vo
        return vi

    def setRobotModelParameters(self, wheel_base):

        self.isModelParameterRetrived_ = True
        self.wheel_base_ = wheel_base

        H = np.array([
            [1, 0, 0],
            [0, 1, self.wheel_base_/2],
            [1, 0, 0],
            [0, 1, -self.wheel_base_/2]
        ])
        self.kinematical_matrix_ = np.linalg.inv((H.transpose() @ H)) @ H.transpose()
        
    def retriveReferenceFromCSV(self, file_name):
        
        node_lists = []

        with open(file_name, newline='') as csvfile:
            rows = csv.reader(csvfile)
            n = 0
            for row in rows:
                if n == 0:
                    n += 1
                    continue

                node_lists.append([float(e) for e in row])
            # Node List Info:
            # x | y | yaw | vx | vy | w | front_wheel: angle | front_wheel: speed | rear_wheel: angle | rear_wheel: speed  | time_stamp

        self.isReferenceRetrived_ = True
        self.reference_ = node_lists
    
    def getReferenceTrajectoryWithinHorizon(self, x_current):

        if not self.isReferenceRetrived_:
            return None
        
        # Prune already passed path.
        idx = 0
        toPrune = False
        for i in range(len(self.reference_)):

            dist = math.sqrt((x_current[0] - self.reference_[i][0])**2 + (x_current[1] - self.reference_[i][1])**2)
            if dist < self.trim_dist_:
                toPrune = True
                idx = i

        if toPrune:
            del self.reference_[:idx+1]

        # Prepare ref path
        travel = 0
        considered_nodes = []
        for i in range(len(self.reference_)):
            if not considered_nodes:
                considered_nodes.append(self.reference_[i])
                continue
            dx = self.reference_[i][0] - self.reference_[i-1][0]
            dy = self.reference_[i][1] - self.reference_[i-1][1]

            travel = travel + math.sqrt(dx**2 + dy**2)
            if travel <= self.LOOKAHEAD_DIST_ :
                considered_nodes.append(self.reference_[i])
            else:
                break
    
        x_ref = np.zeros((self.NX_, self.HL_ + 1))
        for i in range(self.HL_ + 1):
            j = self.HL_ - i 
            node = considered_nodes[-1]
            x_ref[0, j] = node[0]
            x_ref[1, j] = node[1]
            x_ref[2, j] = node[2]
            if not len(considered_nodes) == 1:
                considered_nodes.pop()

        # print(x_ref)
        return x_ref
        
    def updateCurrentState(self, x, y, yaw):
        x_current = np.zeros((self.NX_, 1))
        x_current[0, 0] = x
        x_current[1, 0] = y
        x_current[2, 0] = yaw
        return x_current
    
    def getRobotModelMatrice(self, theta, v_f, delta_f, v_r, delta_r):

        A = np.zeros((self.NX_, self.NX_))
        B = np.zeros((self.NX_, self.NU_))
        C = np.zeros(self.NX_)
        # Define A
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[0, 2] = -0.5*self.DT_*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.cos(theta) -0.5*self.DT_*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.cos(theta)
        A[1, 2] = 0.5*self.DT_*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.sin(theta) -0.5*self.DT_*(v_f*math.cos(delta_f)+v_r*math.cos(delta_r))*math.sin(theta)
        # Define B
        B[0, 0] =  1 / 2 * self.DT_ * (math.cos(delta_f) * math.cos(theta) - math.sin(delta_f) * math.sin(theta))
        B[0, 1] = -1 / 2 * self.DT_ * (math.sin(delta_f) * math.cos(theta) + math.cos(delta_f) * math.sin(theta)) * v_f
        B[0, 2] =  1 / 2 * self.DT_ * (math.cos(delta_r) * math.cos(theta) - math.sin(delta_r) * math.sin(theta))
        B[0, 3] = -1 / 2 * self.DT_ * (math.sin(delta_r) * math.cos(theta) + math.cos(delta_r) * math.sin(theta)) * v_r
        B[1, 0] =  1 / 2 * self.DT_ * (math.cos(delta_f) * math.sin(theta) + math.sin(delta_f) * math.cos(theta))
        B[1, 1] = -1 / 2 * self.DT_ * (math.sin(delta_f) * math.sin(theta) - math.cos(delta_f) * math.cos(theta)) * v_f
        B[1, 2] =  1 / 2 * self.DT_ * (math.cos(delta_r) * math.sin(theta) + math.sin(delta_r) * math.cos(theta))
        B[1, 3] = -1 / 2 * self.DT_ * (math.sin(delta_r) * math.cos(theta) - math.cos(delta_r) * math.sin(theta)) * v_r
        B[2, 0] = -1 / self.wheel_base_ * self.DT_ * math.sin(delta_r)
        B[2, 1] = -1 / self.wheel_base_ * self.DT_ * math.cos(delta_f) * v_f 
        B[2, 2] = -1 / self.wheel_base_ * self.DT_ * math.sin(delta_r)
        B[2, 3] = -1 / self.wheel_base_ * self.DT_ * math.cos(delta_r) * v_r
        # Define C
        C[0] = 1 / 2 * self.DT_ * (v_f * math.cos(delta_f) * math.sin(theta) * (delta_f + theta) + v_r * math.cos(delta_r) * math.sin(theta) * (delta_r + theta) + v_f * math.sin(delta_f) * math.cos(theta) * (delta_f + theta) + v_r * math.sin(delta_r) * math.cos(theta) * (delta_r + theta))
        C[1] = 1 / 2 * self.DT_ * (-v_f * math.cos(delta_f) * math.cos(theta) * (delta_f + theta) - v_r * math.cos(delta_r) * math.cos(theta) * (delta_r + theta) + v_f * math.sin(delta_f) * math.sin(theta) * (delta_f + theta) + v_r * math.sin(delta_r) * math.sin(theta) * (delta_r + theta))
        C[2] = 1 / self.wheel_base_ * self.DT_ * (-math.cos(delta_f) * delta_f * v_f + math.cos(delta_r) * delta_r * v_r)
        
        return A, B, C
  
    def predictMotion(self, x_ref, x_current, control_input):
        
        """
        x_ref         : array, size = (NX, HL+1)
        x_current     : array, size = (NX,    1)
        control_input : list,  size = [HL][NU]
        """

        # make an empty array
        x_predicted = x_ref * 0.0
        # set init state
        x_predicted[0, 0] = x_current[0, 0]
        x_predicted[1, 0] = x_current[1, 0]
        x_predicted[2, 0] = x_current[2, 0]
        # predict future state 
        for i in range(len(control_input)):

            V = self.getVelocitiesFromRobotModels(
                front_speed=control_input[i][0], 
                front_steer=control_input[i][1], 
                rear_speed=control_input[i][2],
                rear_steer=control_input[i][3] 
            )
            
            x_predicted[0, i+1] = x_predicted[0, i] + V[0] * math.cos(x_predicted[2, i]) * self.DT_ - V[1] * math.sin(x_predicted[2, i]) * self.DT_
            x_predicted[1, i+1] = x_predicted[1, i] + V[0] * math.sin(x_predicted[2, i]) * self.DT_ + V[1] * math.cos(x_predicted[2, i]) * self.DT_
            x_predicted[2, i+1] = x_predicted[2, i] + V[2] * self.DT_

        return x_predicted




if __name__ == '__main__' :

    MPC = ModelPredictiveControl()
    MPC.initialization(
        wheel_base=1,
        file_name="reference.csv"
    )
    MPC.start()
    # print(MPC.reference_[0])

