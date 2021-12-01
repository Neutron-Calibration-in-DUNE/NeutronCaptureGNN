"""

"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plot_capture(input_file):
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
        ax.scatter(clusters[i][0],clusters[i][1],clusters[i][2],s=50,label=sum(clusters[i][3]))
    ax.set_title(total_energy)
    plt.legend()
    plt.show()

plot_capture("../data/raw/train_9.csv")