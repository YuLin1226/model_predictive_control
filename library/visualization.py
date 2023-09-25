import matplotlib.pyplot as plt
import numpy as np
import math

class CarViz:

    def __init__(self, car_size=[1.5, 0.6]) -> None:
        
        self.length_ = car_size[0]
        self.width_ = car_size[1]

    def plotCar(self, x, y, yaw, truckcolor="-k"): 
    
        outline = np.array([[-self.length_/2, self.length_/2, self.length_/2, -self.length_/2, -self.length_/2],
                            [self.width_ / 2, self.width_ / 2, - self.width_ / 2, -self.width_ / 2, self.width_ / 2]])

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])

        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y

        plt.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), truckcolor)
        
    def vizOn(self):
        plt.show()

    def plotTrajectory(self, cx, cy):
        plt.plot(cx, cy, "k-")


    def showAnimation(self, ox, oy, cx, cy, x, y, xref, target_ind, state):

        # plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        if ox is not None:
            plt.plot(ox, oy, "xr", label="MPC")
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(x, y, "ob", label="trajectory")
        plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
        plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        # self.plotCar(state.x, state.y, state.yaw)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)