# %%
# Imports

import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget

# %%
# Input data
dim = 2  # for 2D plots I need dim=2
n_train_samples = 100

centre_positive_samples = np.array([2.0, 6.0]).reshape(-1, 2)
centre_negative_samples = np.array([6.0, 2.0]).reshape(-1, 2)


epsilon = 0.1
# %%
# Define training dataset
n_positive_samples = np.random.randint(1, n_train_samples)
n_negative_samples = n_train_samples - n_positive_samples


# fmt: off
x_positive_samples = np.random.normal(
    loc=centre_positive_samples, 
    scale=(1.0, 1.0), 
    size=(n_positive_samples, dim)
)

x_negative_samples = np.random.normal(
    loc=centre_negative_samples, 
    scale=(1.0, 1.0), 
    size=(n_negative_samples, dim)
)
# fmt: on


# %%
# plot dataset
def plot_train_set_and_w(w_unit=None, n_iterations=None, fig=None, ax=None):
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1, 1)

    # training set
    for x, col in zip([x_positive_samples, x_negative_samples], ["r", "b"]):
        ax.scatter(x[:, 0], x[:, 1], c=col, label=f"n={x.shape[0]}")

    # w vector
    if w_unit is not None:
        # normalise
        w_unit = w / np.linalg.norm(w)

        # normal vector
        ax.quiver(
            0.0,
            0.0,
            w_unit[0],
            w_unit[1],
            angles="xy",
            scale=0.75,
            scale_units="xy",
            color="r",
        )

        # hyperplane: points it goes thru
        ax.axline((0.0, 0.0), (w_unit[1], -w_unit[0]), color="r")

    # title
    if n_iterations:
        ax.set_title(f"Iteration: {n_iterations}")

    # axes
    
    # ax.set_xlim(-5, None)
    # ax.set_ylim(-5, None)
    ax.axis("equal")

    return fig, ax


fig, ax = plot_train_set_and_w()

# %%
# build full dataset
x_positive_samples_w_label = np.hstack(
    [x_positive_samples, np.ones((x_positive_samples.shape[0], 1))]
)

x_negative_samples_w_label = np.hstack(
    [x_negative_samples, -np.ones((x_negative_samples.shape[0], 1))]
)

train_dataset = np.vstack([x_positive_samples_w_label, x_negative_samples_w_label])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Weight update rules


def w_update_naive(w, x):
    return w + x[:2] * x[2]


def w_update_cross(w, x):

    if np.all(w == np.zeros_like(w)):
        w += epsilon


    # unit vectors
    w = w / np.linalg.norm(w)
    x = x / np.linalg.norm(x)

    # compute normal to the plane pi, defined by w and x
    n = np.cross(
        np.pad(w, (0, 1)),
        np.pad(x[:2], (0, 1)),  # np.pad(x[:2] * x[2], (0,1))  # 
    ) # 3D

    # compute vector contained in the plane pi and perpendicular to x
    # will always point to the same semiplane as w?
    w_perp = np.cross(x[:2], n) # 3d # w_perp = np.cross(x[:2]* x[2], n) # 

    # Compute vector to add to w, st the resulting normal is w_perp
    d = w_perp[:2] - w

    return w + 1.1*d


# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Implement naive perceptron

# initialise w
w = np.zeros_like(train_dataset[0, :2]) # modified to non-zero if using w_update_cross
fig, ax = plot_train_set_and_w(w)

n_iterations = 0 # n of weight updates
while n_iterations < 50:  #
    n_misclassif = 0

    for x in train_dataset:
        # check classification
        if (np.dot(w, x[:2]) * x[2] > 0):
            continue
        else:
            # Update rule
            # w += x[:2] * x[2]
            w = w_update_cross(w, x)  # w_update_naive(w, x) #

            # counters
            n_misclassif += 1
            n_iterations += 1

            # plot update
            fig, ax = plot_train_set_and_w(w, n_iterations)

    if n_misclassif == 0:
        break


w_unit = w / np.linalg.norm(w)
n_iterations


# %%
# Check results

proj_x_on_w = (w.reshape(1,2)@train_dataset[:,:2].T).T

plt.figure()
plt.plot(proj_x_on_w, '.-')
plt.hlines(0,xmin=0, xmax=proj_x_on_w.shape[0])
plt.title('Projection of x on w')


# %%
