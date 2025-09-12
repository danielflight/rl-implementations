from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

style.use('ggplot')

def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3
    
def plot_qtables(rundir = None, num_episodes = 25000, algorithm = 'qlearning', create_vid = True):
    """
    Plots the qtables at each episode and, optionally, creates a video.

    from https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/
    """
    
    if not rundir:
        rundir = os.getcwd()
    fig = plt.figure(figsize=(12, 9))

    for i in range(0, num_episodes, 10):
        print(i)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        q_table = np.load(f"{rundir}/qtables/{algorithm}/{i}-qtable.npy")

        for x, x_vals in enumerate(q_table):
            for y, y_vals in enumerate(x_vals):
                ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
                ax2.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
                ax3.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

                ax1.set_ylabel("Action 0")
                ax2.set_ylabel("Action 1")
                ax3.set_ylabel("Action 2")

        #plt.show()
        os.makedirs(f"{rundir}/qtable_charts", exist_ok=True)
        plt.savefig(f"{rundir}/qtable_charts/{i}.png")
        plt.clf()

    if create_vid:
        # def make_video():
        # windows:
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Linux:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))

        for i in range(0, 14000, 10):
            img_path = f"{rundir}/qtable_charts/{i}.png"
            print(img_path)
            frame = cv2.imread(img_path)
            out.write(frame)

        out.release()
        # make_video()
