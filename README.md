Task 1
============

Problem formulation
-------------------

The goal of the problem is to predict the charges $c_1, c_2$ and $c_3$
of the particles with a fixed position given the trajectory of other
particle moving in the 2D space which depend on the above-mentioned
charges. The trajectory data is provided in form of a sequence of
positions of each of the particles in the space with a fixed time step
between those positions. Formally the data-set contains the initial
trajectory data, with each data point being a trajectory
$D = \{\textbf{x}_1, \textbf{x}_2 ... \textbf{x}_n\}$, where
$\textbf{x}_t є \textbf{R}^2$ is a coordinates of particle in the 2D
space at time $t$. The labels for each data-point are given in form
$T = \{c_1, c_2, c_3\}$, where $c_1, c_2$ and $c_3$ are the
corresponding charges.

The problem described is a supervised learning problem. Based on the
given sequence, and the corresponding label we have to perform a
sequence classification. Classification, is because even though the
output data contains a couple data points, they are not sequential and
independent of each-other so all can be viewed all can be represented as
a single prediction.

To evaluate the model, we will simply use the L1 loss, which is simply
the sum of euclidean distances between the predicted data and the actual
data: $$L = \sum_{i=0}^{N} |y - \hat{y}|$$ where $y$ is the ground truth
$\hat{y}$ is the predicted values and $N$ size of the prediction (in our
case 3 charges).

The above loss function, is also easily interpret-able, and could be
used as measure of the model performance.

Task 2
============

Problem formulation
-------------------

The goal of the task is, given the $100\pm10$ steps of initial
simulation, predict the continuation of the $40\pm20$ steps. The input
data is the same as in assignment 3.1, however the ground truth for
output data contains the continuation of the particles trajectory:
$D_c = \{\textbf{x}_{n+1}, \textbf{x}_{n+2} ... \textbf{x}_{n+m}\}$,
where $n$ is the length of the input sequence in steps, $m$ is the
length of the output sequence in steps and
$\textbf{x}_t \in \mathbf{R}^2$ is the position of the particle in 2D
space at time $t$.

The given problem is, again, a supervised learning problem. However, in
contrary to the previous subtask (3.1), in subtask 3.2 the goal is to
predict the sequence based on the other sequence, so it is a Sequence to
Sequence problem.

In this case it is desirable to use the previous values for predicting
the successive ones while it is clear and logical that the successive
values are dependent on the previous ones (the next position of the
particle is partially derived from the previous position).

The simple way to measure the loss of this model would be to look at the
average distance between the predicted position of the particle and it's
true position at that time. We use mean of the Pairwise-Distance
function for it (because L1 loss takes the distance element-wise rather
than pairwise).
$$L = \frac{1}{N}\sum_{t=0}^{N} |\textbf{y}_t - \hat{\textbf{y}}_t|$$

where N is the length of output sequence, $y_t$ is the ground truth at
time t (position of the particle), and $\hat{y}_t$ is the predicted
position of the particle at time t.

This loss function is easily interpret-able and at the same time
intuitively representative of the goal of finding the curves that are
close together.

There is however one improved version of such loss. Rather than just
looking at the distance between the predicted trajectories, we could
look at the difference between their velocities. That way, we would
incorporate some more logic of the simulation into the network, and help
the network, by decreasing the loss function, if the two trajectories
are relatively distant, but they're velocities match, which means, their
shape is similar. We can approximate the velocity at time $t$ by
$v_t = \textbf{y}_t - \textbf{y}_{t-1}$, and therefore have the
following formula for loss:
$$L = \frac{\sum_{t=0}^{N} |\textbf{y}_t - \hat{\textbf{y}}_t| + \sum_{i=0}^{N} |(\textbf{y}_t - \textbf{y}_{t-1}) - (\hat{\textbf{y}}_{t} - \hat{\textbf{y}}_{t-1})|}{2N}$$

This loss is less interpret-able and is not suitable for using as a
performance metric of the Neural Network. However, supposedly, is more
effective for training.
