"""

"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from itertools import permutations
from random import sample, shuffle
import argparse
import os
import sys
sys.path.append("../")
sys.path.append("../../")

#Get the color-wheel
Nlines = 200
color_lvl = 8
colors = np.array(list(permutations(range(0,256,color_lvl),3)))/255.0
np.random.shuffle(colors)


def plot_capture(input_file,
    save='',
    show=False
):
    x = []
    y = []
    z = []
    gamma = []
    energy = []
    clusters = []
    with open(input_file, "r") as file:
        reader = csv.reader(file,delimiter=",")
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))
            energy.append(float(row[4]))
            gamma.append(int(row[5]))
    print(np.unique(gamma))
    temp_gamma = gamma[0]
    temp_cluster = [[],[],[],[]]
    for i in range(len(x)):
        if gamma[i] == temp_gamma:
            temp_cluster[0].append(x[i])
            temp_cluster[1].append(y[i])
            temp_cluster[2].append(z[i])
            temp_cluster[3].append(energy[i])
        else:
            clusters.append(temp_cluster)
            temp_gamma = gamma[i]
            temp_cluster = [[x[i]],[y[i]],[z[i]],[energy[i]]]
        if i == len(x)-1:
            clusters.append(temp_cluster)
    total_energy = sum(energy)
    print(len(clusters))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    # plot the starting position
    for i in range(len(clusters)):
        for j in range(len(clusters[i][0])):
            if j == 0:
                ax.scatter(clusters[i][0][j],clusters[i][1][j],clusters[i][2][j],s=clusters[i][3][j]*1000,color=colors[i],label=sum(clusters[i][3]))
            else:
                ax.scatter(clusters[i][0][j],clusters[i][1][j],clusters[i][2][j],s=clusters[i][3][j]*1000,color=colors[i])
    ax.set_title(total_energy)
    plt.legend()
    if save != '':
        plt.savefig(save)
    if show == True:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int,dest='n')
    parser.add_argument("save", nargs='?')
    parser.add_argument("show", nargs='?')

    args = parser.parse_args()
    event = args.n
    if 'save' in parser.parse_args(['save']):
        save = f"../../plots/train_{event}.png"
    else:
        save = ''
    if 'show' in parser.parse_args(['show']):
        show = True
    else:
        show = False

    plot_capture(f"../data/raw/train_{event}.csv",
            save=save,show=show)
    