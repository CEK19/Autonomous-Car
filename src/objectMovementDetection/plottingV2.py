"""
Plot tools 2D
@author: Minh The Tus
"""
from const import *
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")


class Plotting:
    def __init__(self, xI, xG, obs):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        self.xI, self.xG = xI, xG
        self.obs = obs              # .obs_map()


    def update_obs(self, obs):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        self.obs = obs


    def animation(self, path, visited, name):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        self.plot_grid(name)
        self.plot_visited(visited)
        self.plot_path(path)
        plt.show()


    def animation_lrta(self, path, visited, name):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        self.plot_grid(name)
        cl = self.color_list_2()
        path_combine = []

        for k in range(len(path)):
            self.plot_visited(visited[k], cl[k])
            plt.pause(0.2)
            self.plot_path(path[k])
            path_combine += path[k]
            plt.pause(0.2)
        if self.xI in path_combine:
            path_combine.remove(self.xI)
        self.plot_path(path_combine)
        plt.show()


    def animation_ara_star(self, path, visited, name):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        self.plot_grid(name)
        cl_v, cl_p = self.color_list()

        for k in range(len(path)):
            self.plot_visited(visited[k], cl_v[k])
            self.plot_path(path[k], cl_p[k], True)
            plt.pause(0.5)

        plt.show()


    def animation_bi_astar(self, path, v_fore, v_back, name):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        self.plot_grid(name)
        self.plot_visited_bi(v_fore, v_back)
        self.plot_path(path)
        plt.show()


    def show(self, block=True):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        plt.show(block=block)

    def pause(self, time):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        plt.pause(time)
        
    def clearAll(self):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        plt.cla()
    
    def setTitle(self, title):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        plt.title(title)
    
    def createFig(self):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        return plt.figure()
    
    def draw_idle(self, fig):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        fig.canvas.draw_idle()
    

    def plot_point(self, x, y, option):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        if option == 'addObs':
            plt.plot(x, y, 'sk')
        else:
            plt.plot(x, y, marker='s', color='white')


    def plot_grid(self, name):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")


    def plot_visited(self, visited, cl='gray'):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        if self.xI in visited:
            visited.remove(self.xI)

        if self.xG in visited:
            visited.remove(self.xG)

        count = 0

        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o')
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40
            #
            # length = 15

            if count % length == 0:
                plt.pause(0.001)
        plt.pause(0.01)


    def plot_path(self, path, cl='r', flag=False):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        if not flag:
            plt.plot(path_x, path_y, linewidth='3', color='r')
        else:
            plt.plot(path_x, path_y, linewidth='3', color=cl)

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

        plt.pause(0.01)


    def plot_visited_bi(self, v_fore, v_back):
        if not D_STAR.ENV.IS_PLOTTING:
            return
        if self.xI in v_fore:
            v_fore.remove(self.xI)

        if self.xG in v_back:
            v_back.remove(self.xG)

        len_fore, len_back = len(v_fore), len(v_back)

        for k in range(max(len_fore, len_back)):
            if k < len_fore:
                plt.plot(v_fore[k][0], v_fore[k][1],
                         linewidth='3', color='gray', marker='o')
            if k < len_back:
                plt.plot(v_back[k][0], v_back[k][1], linewidth='3',
                         color='cornflowerblue', marker='o')

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 10 == 0:
                plt.pause(0.001)
        plt.pause(0.01)


    @staticmethod
    def color_list():
        cl_v = ['silver',
                'wheat',
                'lightskyblue',
                'royalblue',
                'slategray']
        cl_p = ['gray',
                'orange',
                'deepskyblue',
                'red',
                'm']
        return cl_v, cl_p


    @staticmethod
    def color_list_2():
        cl = ['silver',
              'steelblue',
              'dimgray',
              'cornflowerblue',
              'dodgerblue',
              'royalblue',
              'plum',
              'mediumslateblue',
              'mediumpurple',
              'blueviolet',
              ]
        return cl
