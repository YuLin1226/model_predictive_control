import math
import csv

class TrajectoryGenerator:

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
                    if i == 11:
                        node.append(element)
                    else:
                        node.append(float(element))
                node_lists.append(node)
        return node_lists

    def interpolateReference(self, node_lists, interpolate_num=5, mode='ackermann'):

        pts_x, pts_y, pts_yaw = [], [], []
        if mode == 'ackermann' or mode == 'diff':
            for i in range(len(node_lists) - 1):
                vx = node_lists[i+1][3]
                vy = node_lists[i+1][4]
                w  = node_lists[i+1][5]
                if w == 0:
                    continue
                from_node = node_lists[i]
                to_node   = node_lists[i+1]
                for i in range(interpolate_num):
                    icr = [-vy / w, vx / w]
                    yaw = from_node[2] + (to_node[2] - from_node[2]) / interpolate_num * i
                    x = (math.cos(from_node[2]) - math.cos(yaw)) * icr[0] - (math.sin(from_node[2]) - math.sin(yaw)) * icr[1] + from_node[0]
                    y = (math.sin(from_node[2]) - math.sin(yaw)) * icr[0] + (math.cos(from_node[2]) - math.cos(yaw)) * icr[1] + from_node[1]
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_yaw.append(yaw)
            return pts_x, pts_y, pts_yaw
        
        if mode == 'crab':
            for i in range(len(node_lists) - 1):
                vx = node_lists[i+1][3]
                vy = node_lists[i+1][4]
                w  = node_lists[i+1][5]
                from_node = node_lists[i]
                to_node   = node_lists[i+1]
                for i in range(interpolate_num):
                    yaw = from_node[2]
                    x = from_node[0] + (to_node[0] - from_node[0]) / interpolate_num * i
                    y = from_node[1] + (to_node[1] - from_node[1]) / interpolate_num * i
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_yaw.append(yaw)
            return pts_x, pts_y, pts_yaw
        
    def interpolateReference2(self, node_lists, interpolate_num=5):

        pts_x, pts_y, pts_yaw, pts_mode = [], [], [], []
        for i in range(len(node_lists) - 1):
            mode = node_lists[i+1][11]
            if mode == 'ackermann' or mode == 'diff':
                vx = node_lists[i+1][3]
                vy = node_lists[i+1][4]
                w  = node_lists[i+1][5]
                if w == 0:
                    continue
                from_node = node_lists[i]
                to_node   = node_lists[i+1]
                for i in range(interpolate_num):
                    icr = [-vy / w, vx / w]
                    yaw = from_node[2] + (to_node[2] - from_node[2]) / interpolate_num * i
                    x = (math.cos(from_node[2]) - math.cos(yaw)) * icr[0] - (math.sin(from_node[2]) - math.sin(yaw)) * icr[1] + from_node[0]
                    y = (math.sin(from_node[2]) - math.sin(yaw)) * icr[0] + (math.cos(from_node[2]) - math.cos(yaw)) * icr[1] + from_node[1]
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_yaw.append(yaw)
                    pts_mode.append(mode)
            
            elif mode == 'crab':
                vx = node_lists[i+1][3]
                vy = node_lists[i+1][4]
                w  = node_lists[i+1][5]
                from_node = node_lists[i]
                to_node   = node_lists[i+1]
                for i in range(interpolate_num):
                    yaw = from_node[2]
                    x = from_node[0] + (to_node[0] - from_node[0]) / interpolate_num * i
                    y = from_node[1] + (to_node[1] - from_node[1]) / interpolate_num * i
                    pts_x.append(x)
                    pts_y.append(y)
                    pts_yaw.append(yaw)
                    pts_mode.append(mode)

        return pts_x, pts_y, pts_yaw, pts_mode

    def removeRepeatedPoints(self, cx, cy, cyaw, epsilon=0.00001):

        nx, ny, nyaw = [], [], []
        for x, y, yaw in zip(cx, cy, cyaw):
            if not nx:
                nx.append(x)
                ny.append(y)
                nyaw.append(yaw)
                continue
            dx = x - nx[-1]
            dy = y - ny[-1]
            if (dx**2 + dy**2) < epsilon:
                continue
            nx.append(x)
            ny.append(y)
            nyaw.append(yaw)
        return nx, ny, nyaw

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

    def makeEightShapeTrajectory(self, size=10, n=121):
        x, y, yaw = [], [], []
        for i in range(n):
            ptx = 0.8 * math.sin(2 * math.pi / 60 * i) * size
            pty = math.sin(1 * math.pi / 60 * i) * size
            dx = 0.8 * math.cos(2 * math.pi / 60 * i) * 2 * math.pi / 60
            dy = math.cos(1 * math.pi / 60 * i) * 1 * math.pi / 60
            ptyaw = math.atan2(dy, dx)
            x.append(ptx)
            y.append(pty)
            yaw.append(ptyaw)
        return x ,y, yaw