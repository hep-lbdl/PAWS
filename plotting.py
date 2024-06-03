import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import os

from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D

sea.set(style="white")

#plot landscape dynamically
def loss_landscape_nofit(sigfrac, m1, m2, z, step=0.25, save = False, decay = "qq"):
    start = 0.5
    end = 6
    step = step
    
    weight_list = np.arange(start, end + step, step)
    grid_axes = [(w1, w2) for w1 in weight_list for w2 in weight_list]
    w1_values, w2_values = zip(*grid_axes)

    loss_values = list(z[sigfrac, m1, m2, decay])
    min_loss = min(loss_values)
    max_loss = max(loss_values)
    normalized_loss = [(x - min_loss) / (max_loss - min_loss) for x in loss_values]
    bins = int(np.sqrt(len(z[sigfrac, m1, m2])))

    star1_coords = (m1, m2)
    star2_coords = (m2, m1)

    plt.figure(figsize=(8, 6))
    h = plt.hist2d(w1_values, w2_values, bins=(bins, bins), cmap='viridis', weights=normalized_loss)
    plt.scatter(*star1_coords, c='red', marker='*', s=200, label='Star 1')
    plt.scatter(*star2_coords, c='blue', marker='*', s=200, label='Star 2')
    plt.colorbar(label='Loss (BCE)')
    plt.xlabel('m1')
    plt.ylabel('m2')
    plt.title('6 Features (m1 = {} | m2 = {}) sigfrac: {:.4f}'.format(m1, m2, sigfrac))
    plt.legend()
    plt.show()
    
    if save == True:
        plt.savefig(f'plots/landscape{float(m1)}{float(m2)}_{decay}.png', dpi=450, bbox_inches='tight')
    
    return h

#Loss Landscape but 3D
#change elv and azim for viewing angle
#step is resolution
def create_3D_loss_manifold(sigfrac, m1, m2, z, step, elev, azim, save = False, decay = "qq"):

    start = 0.5
    end = 6
    step = step

    weight_list = np.arange(start, end + step, step)

    grid_axes = []
    for w1 in weight_list:
        for w2 in weight_list:
            grid_axes.append((w1, w2))

    w1_values, w2_values = zip(*grid_axes)

    loss_values = list(z[sigfrac, m1, m2, decay])

    x = w1_values
    y = w2_values
    z = loss_values

    sea.set(style="whitegrid")
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='.', alpha = 0.1)
    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('W1')
    ax.set_ylabel('W2')
    ax.set_zlabel('Loss')
    ax.set_title(f"Loss Manifold m1: {m1} m2: {m2} sigfrac: {np.round(sigfrac, 4)}")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)
    
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_zticks([])
    
    ax.view_init(elev=elev, azim=azim)
    
    if save == True:
        plt.savefig(f'plots/manifold{float(m1)}{float(m2)}_{decay}.png', dpi=450, bbox_inches='tight')
    return ax

#interpolating using scipy RectBivariateSpline
def plot_interpolated_landscape(sigfrac, m1, m2, z, step, decay = "qq", save = False):

    start = 0.5
    end = 6
    step = step

    weight_list = np.arange(start, end + step, step)

    x_values = weight_list
    y_values = weight_list

    x, y = np.meshgrid(x_values, y_values)

    #loss_values_flat = np.array(z[1e-6, 5, 1, "qq"])[:,0]
    loss_values_flat = z[sigfrac, m1, m2, decay]
    loss_values = np.array(loss_values_flat).reshape(x.shape)

    interp_spline = RectBivariateSpline(x_values, y_values, loss_values, s = 0)

    xi, yi = np.meshgrid(np.linspace(min(x_values), max(x_values), 1000), np.linspace(min(y_values), max(y_values), 1000))
    zi = interp_spline(xi[0, :], yi[:, 0])
    
    #3d projection
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', linewidth = 0)
    ax3d.set_title("Loss Manifold")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False

    ax3d.grid(False)
    
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax3d.set_zticks([])
    ax3d.view_init(elev=70, azim=250)

    #2d projection
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    h0 = ax[0].pcolormesh(x, y, loss_values, cmap='viridis')
    ax[0].set_aspect("equal")
    ax[0].set_title(f"$m_{1} = {m1} \quad m_{2} = {m2} \quad$" + f"sigfrac ={sigfrac}")
    ax[0].set_xlabel(r"$w_{1}$")
    ax[0].set_ylabel(r"$w_{2}$")
    cbar = plt.colorbar(h0, ax=ax[0])    
    
    h1 = ax[1].contourf(xi, yi, zi, cmap='viridis')
    ax[1].set_aspect("equal")
    ax[1].set_title(f"$m_{1} = {m1} \quad m_{2} = {m2} \quad$" + f"sigfrac ={sigfrac}")
    ax[1].set_xlabel(r"$w_{1}$")
    ax[1].set_ylabel(r"$w_{2}$")
    cbar = plt.colorbar(h1, ax=ax[1])
    
    plt.tight_layout()
    if save == True:
        plt.savefig(f'plots/interpolation{float(m1)}{float(m2)}.png', dpi=450, bbox_inches='tight')

def AUC_landscape_nofit(sigfrac, m1, m2, a, step=0.25, save = False):
    start = 0.5
    end = 6
    step = step
    
    weight_list = np.arange(start, end + step, step)
    grid_axes = [(w1, w2) for w1 in weight_list for w2 in weight_list]
    w1_values, w2_values = zip(*grid_axes)

    loss_values = list(a[sigfrac, m1, m2])
    min_loss = min(loss_values)
    max_loss = max(loss_values)
    normalized_loss = [(x - min_loss) / (max_loss - min_loss) for x in loss_values]
    bins = int(np.sqrt(len(z[sigfrac, m1, m2])))

    star1_coords = (m1, m2)
    star2_coords = (m2, m1)

    plt.figure(figsize=(8, 6))
    h = plt.hist2d(w1_values, w2_values, bins=(bins, bins), cmap='viridis', weights=normalized_loss)
    plt.scatter(*star1_coords, c='red', marker='*', s=200, label='Star 1')
    plt.scatter(*star2_coords, c='blue', marker='*', s=200, label='Star 2')
    plt.colorbar(label='Loss (BCE)')
    plt.xlabel('m1')
    plt.ylabel('m2')
    plt.title('6 Features (m1 = {} | m2 = {}) sigfrac AUC Landscape: {:.4f}'.format(m1, m2, sigfrac))
    plt.legend()
    plt.show()
    
    if save == True:
        plt.savefig(f'plots/landscapes/AUCL_{float(m1)}{float(m2)}.png', dpi=450, bbox_inches='tight')
    
    return h

def plot_landscapes(sigfrac, m1, m2, z, a, step, save = False):
    start = 0.5
    end = 6
    step = step

    #weightspace
    weight_list = np.arange(start, end + step, step)
    grid_axes = [(w1, w2) for w1 in weight_list for w2 in weight_list]
    w1_values, w2_values = zip(*grid_axes)

    #loss
    loss_values = list(z[sigfrac, m1, m2])
    min_loss = min(loss_values)
    max_loss = max(loss_values)
    
    AUC_values = list(a[sigfrac, m1, m2])
    min_loss = min(AUC_values)
    max_loss = max(AUC_values)
    bins = int(np.sqrt(len(a[sigfrac, m1, m2])))

    star1_coords = (m1, m2)
    star2_coords = (m2, m1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist2d(w1_values, w2_values, bins = (bins, bins), cmap='viridis', weights=loss_values)
    ax[0].set_aspect("equal")
    ax[0].set_title(f"Loss Landscape")
    ax[0].set_xlabel(r"$w_{1}$")
    ax[0].set_ylabel(r"$w_{2}$")
    ax[0].scatter(*star1_coords, c='red', marker='*', s=200, label='Star 1')
    ax[0].scatter(*star2_coords, c='blue', marker='*', s=200, label='Star 2')
    
    ax[1].hist2d(w1_values, w2_values, bins = (bins, bins), cmap='viridis', weights=AUC_values)
    ax[1].set_aspect("equal")
    ax[1].set_title("AUC Landscape")
    ax[1].set_xlabel(r"$w_{1}$")
    ax[1].set_ylabel(r"$w_{2}$")
    ax[1].scatter(*star1_coords, c='red', marker='*', s=200, label='Star 1')
    ax[1].scatter(*star2_coords, c='blue', marker='*', s=200, label='Star 2')

    plt.subplots_adjust(top=0.95)
    #plt.tight_layout()
    fig.suptitle(f"$m_{1}: ${m1 * 100} $m_{2}: {m2 * 100}$")
    if save == True:
        plt.savefig(f"plots/bothlandscape{float(m1)}{float(m2)}.png", dpi=450, bbox_inches='tight')
    return ax