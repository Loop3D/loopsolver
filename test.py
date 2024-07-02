from LoopStructural import GeologicalModel
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import logging
import pyvista as pv
from LoopStructural.interpolators.supports import StructuredGrid, TetMesh
from LoopStructural.interpolators import FiniteDifferenceInterpolator, P1Interpolator

origin = [0, 0, 0]
maximum = np.array([1000, 1000, 1000])

p1 = np.array(
    np.meshgrid(
        np.linspace(origin[0], maximum[0] / 2 - 100, 10),
        np.linspace(origin[1], maximum[1] / 2 - 100, 10),
        np.linspace(origin[2], maximum[2] / 2 - 100, 10),
        indexing="ij",
    )
).T.reshape(-1, 3)
p2 = np.array(
    np.meshgrid(
        np.linspace(maximum[0] / 2, maximum[0], 10),
        np.linspace(maximum[1] / 2, maximum[1], 10),
        np.linspace(maximum[2] / 2, maximum[2], 10),
        indexing="ij",
    )
).T.reshape(-1, 3)

p1 = np.random.random((100, 3)) * maximum / 2
p2 = np.random.random((100, 3)) * maximum / 2 + maximum / 2


nsteps = np.array([10, 10, 10])
step_vector = np.array(maximum) / nsteps
# support = StructuredGrid(origin, step_vector=step_vector, nsteps=nsteps)
support = TetMesh(origin, step_vector=step_vector, nsteps=nsteps)
p1 = np.hstack([p1, np.zeros((p1.shape[0], 2))])
p1[:, 3] = -10000
p1[:, 4] = -np.finfo(float).eps
p2 = np.hstack([p2, np.zeros((p2.shape[0], 2))])
p2[:, 4] = 10000
p2[:, 3] = np.finfo(float).eps

results = []
# units = drillpts['id'].values
pts = np.vstack(
    [p1, p2]
)  # np.vstackdrillpts[['X','Y','Z']].values,np.zeros((drillpts.shape[0],2))])

interpolator = P1Interpolator(support)  # FiniteDifferenceInterpolator(support)  #


interpolator.set_value_inequality_constraints(pts)
interpolator.setup_interpolator()
orthogonal_pts = support.barycentre
np.random.shuffle(orthogonal_pts)
orthogonal_pts = orthogonal_pts[:1000]
# interpolator.add_gradient_orthogonal_constraints(orthogonal_pts,np.tile(np.array([-0.00341065,  0.69922801, -0.68977915]),(orthogonal_pts.shape[0],1)),w=.01,B=0 )
# Q, bounds = interpolator.build_inequality_matrix()
# A, b = interpolator.build_matrix()
# v, proc = solve_inequalities(A, b, Q, bounds,support.n_nodes,'test',  admmweight=2,nmajor=200,mode='values', init_values=support.nodes[:,2], tomofast_location="/home/lgrose/Documents/repositories/loop/Tomofast-x/tomofastx",
# )

# interpolator.add_gradient_orthogonal_constraints(
#     orthogonal_pts,
#     np.tile(
#         np.array([-0.00341065, 0.69922801, -0.68977915]), (orthogonal_pts.shape[0], 1)
#     ),
#     w=0.1,
#     B=0,
# )
Q, bounds = interpolator.build_inequality_matrix()
A, b = interpolator.build_matrix()
# v, proc = solve_inequalities(A, b, Q, bounds,support.n_nodes,'test',  admmweight=0.3,mode='values', init_values=v[0], tomofast_location="/home/lgrose/Documents/repositories/loop/Tomofast-x/tomofastx",
# )

from loopsolver import solve


x = solve(A, b, Q, bounds, support.nodes[:, 2], 0.001, support.n_nodes, 1)

vtk_grid = support.vtk
vtk_grid["unit"] = x

from loopstructuralvisualisation import Loop3DView

vals1 = support.evaluate_value(p1[:, :3], x)
vals2 = support.evaluate_value(p2[:, :3], x)
plt.hist(vals1)
# plt.twinx()
plt.hist(vals2)
plt.show()
view = Loop3DView()
view.add_points(p1[:, :3], color="blue")
view.add_points(p2[:, :3], color="red")
view.add_mesh(vtk_grid.contour(10))
view.show()
