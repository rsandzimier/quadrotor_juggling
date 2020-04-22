import pydrake
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import DiagramBuilder, LinearQuadraticRegulator, Simulator, plot_system_graphviz 
from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder,
                         PlanarSceneGraphVisualizer, SceneGraph, TrajectorySource,
                         SnoptSolver, MultibodyPositionToGeometryPose, PiecewisePolynomial,
                         MathematicalProgram, JacobianWrtVariable, eq, le, ge,IpoptSolver, DirectCollocation,
                         InputPortIndex)
from pydrake.autodiffutils import autoDiffToValueMatrix
from quadrotor2d import Quadrotor2D
from ball2d import Ball2D
from visualization import Visualizer

n_quadrotors = 1
n_balls = 0

def makeDiagram(n_quadrotors, n_balls, use_visualizer=False, trajectory=None):
    builder = DiagramBuilder()
    # Setup quadrotor plants and controllers
    quadrotor_plants = []
    quadrotor_controllers = []
    for i in range(n_quadrotors):
        new_quad = Quadrotor2D(n_quadrotors=n_quadrotors-1, n_balls=n_balls)
        new_quad.set_name('quad_' + str(i))
        plant = builder.AddSystem(new_quad)
        if trajectory is None:
            builder.ExportInput(plant.get_input_port(0))
        quadrotor_plants.append(plant)

    # Setup ball plants
    ball_plants = []
    for i in range(n_balls):
        new_ball = Ball2D(n_quadrotors=n_quadrotors, n_balls=n_balls-1)
        new_ball.set_name('ball_' + str(i))
        plant = builder.AddSystem(new_ball)
        ball_plants.append(plant)

    # Connect all plants so that each object (quadrotor or ball) has access to all other object states as inputs
    for i in range(n_quadrotors):
        for j in range(n_quadrotors):
            if i == j: 
                continue
            k = j if j < i else j-1
            builder.Connect(quadrotor_plants[j].get_output_port(0), quadrotor_plants[i].GetInputPort('quad_'+str(k)))
        for j in range(n_balls):
            builder.Connect(ball_plants[j].get_output_port(0), quadrotor_plants[i].GetInputPort('ball_'+str(j)))
    for i in range(n_balls):
        for j in range(n_quadrotors):
            builder.Connect(quadrotor_plants[j].get_output_port(0), ball_plants[i].GetInputPort('quad_'+str(j)))
        for j in range(n_balls):
            if i == j:
                continue
            k = j if j < i else j-1
            builder.Connect(ball_plants[j].get_output_port(0), ball_plants[i].GetInputPort('ball_'+str(k)))

    # Setup visualization
    if use_visualizer:
        visualizer = builder.AddSystem(Visualizer(n_quadrotors=n_quadrotors, n_balls=n_balls))
        for i in range(n_quadrotors):
            builder.Connect(quadrotor_plants[i].get_output_port(0), visualizer.get_input_port(i))
        for i in range(n_balls):
            builder.Connect(ball_plants[i].get_output_port(0), visualizer.get_input_port(n_quadrotors + i))
    # Setup trajectory source
    if trajectory is not None:
        source = builder.AddSystem(TrajectorySource(trajectory))
        builder.Connect(source.get_output_port(0), quadrotor_plants[0].get_input_port(0))

    diagram = builder.Build()
    return diagram

diagram = makeDiagram(n_quadrotors, n_balls, use_visualizer=False)

context = diagram.CreateDefaultContext()
T = 200 #Number of breakpoints
h_min = 0.001
h_max = 0.01
prog = DirectCollocation(diagram, context, T, h_min, h_max)

state_init = np.array([-1.5, -1.0 , -1.1, 0., 0., 0.0])
state_final = np.array([0.0, 0.0, -np.pi/4, 1.0, 1.0, 0.0])

# Initial conditions
prog.AddLinearConstraint(eq(prog.initial_state(), state_init))

# Final conditions
prog.AddLinearConstraint(eq(prog.final_state(), state_final))

# Input constraints
prog.AddConstraintToAllKnotPoints(prog.input()[0]>=0.0)
prog.AddConstraintToAllKnotPoints(prog.input()[0]<=10.0)
prog.AddConstraintToAllKnotPoints(prog.input()[1]>=0.0)
prog.AddConstraintToAllKnotPoints(prog.input()[1]<=10.0)

###############################################################################
# Set up initial guesses
# ??? There is a way to do this for DirectCollocation. See DirectCollocation.SetInitialTrajectory()
# But we currently don't set an initial trajectory

solver = SnoptSolver()
print("Solving...")
result = solver.Solve(prog)
# be sure that the solution is optimal
assert result.is_success()

print(f'Solution found? {result.is_success()}.')

#################################################################################
# Extract results
# get optimal solution
u_opt_poly = prog.ReconstructInputTrajectory(result)
x_opt_poly = prog.ReconstructStateTrajectory(result)

##################################################################################
# Setup diagram for simulation
diagram = makeDiagram(n_quadrotors, n_balls, use_visualizer=True, trajectory=u_opt_poly)
###################################################################################
# Animate
# plt.figure(figsize=(20, 10))
# plot_system_graphviz(diagram)
# plt.show()
# Set up a simulator to run this diagram
simulator = Simulator(diagram)
integrator = simulator.get_mutable_integrator()
integrator.set_maximum_step_size(0.01) # Reduce the max step size so that we can always detect collisions
context = simulator.get_mutable_context()

##############################################3
# # Simulate
duration = x_opt_poly.end_time()
context.SetTime(0.)
context.SetContinuousState(state_init)
simulator.Initialize()
simulator.AdvanceTo(duration)

t_arr = np.linspace(0,duration,100)
context.SetTime(0.)
context.SetContinuousState(state_init)
simulator.Initialize()

# Plot
q_opt = np.zeros((100,6))
q_actual = np.zeros((100,6))
for i in range(100):
    t = t_arr[i]
    simulator.AdvanceTo(t_arr[i])
    q_opt[i,:] = x_opt_poly.value(t).flatten()
    q_actual[i,:] = context.get_continuous_state_vector().CopyToVector()

plt.figure()
plt.plot(t_arr, q_opt[:,:3])
plt.plot(t_arr, q_actual[:,:3])
plt.figure()
plt.plot(t_arr, q_actual[:,:3]-q_opt[:,:3])
plt.show()