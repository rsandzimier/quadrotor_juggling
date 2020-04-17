import pydrake
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import DiagramBuilder, LinearQuadraticRegulator, Simulator, plot_system_graphviz
from quadrotor2d import Quadrotor2D
from ball2d import Ball2D
from visualization import Visualizer

def QuadrotorLQR(plant):
    context = plant.CreateDefaultContext()
    context.SetContinuousState(np.zeros([6, 1]))
    context.FixInputPort(0, plant.mass * plant.gravity / 2. * np.array([1, 1]))

    Q = np.diag([10, 10, 10, 1, 1, (plant.length / 2. / np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]])

    return LinearQuadraticRegulator(plant, context, Q, R)

n_quadrotors = 2
n_balls = 3

builder = DiagramBuilder()
# Setup quadrotor plants and controllers
quadrotor_plants = []
quadrotor_controllers = []
for _ in range(n_quadrotors):
    plant = builder.AddSystem(Quadrotor2D(n_quadrotors=n_quadrotors-1, n_balls=n_balls))
    quadrotor_plants.append(plant)
    controller = QuadrotorLQR(plant)
    quadrotor_controllers = builder.AddSystem(controller)
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

# Setup ball plants
ball_plants = []
for _ in range(n_balls):
    plant = builder.AddSystem(Ball2D(n_quadrotors=n_quadrotors, n_balls=n_balls-1))
    ball_plants.append(plant)

# Connect all plants so that each object (quadrotor or ball) has access to all other object states as inputs
for i in range(n_quadrotors):
    for j in range(n_quadrotors):
        if i == j: 
            continue
        k = j if j < i else j-1
        print ('quad',i,j,k)
        builder.Connect(quadrotor_plants[j].get_output_port(0), quadrotor_plants[i].GetInputPort('quad_'+str(k)))
    for j in range(n_balls):
        print ('ball',i,j)
        builder.Connect(ball_plants[j].get_output_port(0), quadrotor_plants[i].GetInputPort('ball_'+str(j)))
for i in range(n_balls):
    for j in range(n_quadrotors):
        print ('quad',i,j)
        builder.Connect(quadrotor_plants[j].get_output_port(0), ball_plants[i].GetInputPort('quad_'+str(j)))
    for j in range(n_balls):
        if i == j:
            continue
        k = j if j < i else j-1
        print ('ball',i,j,k)
        builder.Connect(ball_plants[j].get_output_port(0), ball_plants[i].GetInputPort('ball_'+str(k)))

# Setup visualization
visualizer = builder.AddSystem(Visualizer(n_quadrotors=n_quadrotors, n_balls=n_balls))
for i in range(n_quadrotors):
    builder.Connect(quadrotor_plants[i].get_output_port(0), visualizer.get_input_port(i))
for i in range(n_balls):
    builder.Connect(ball_plants[i].get_output_port(0), visualizer.get_input_port(n_quadrotors + i))

diagram = builder.Build()

# plt.figure(figsize=(20, 10))
# plot_system_graphviz(diagram)
# plt.show()
# Set up a simulator to run this diagram
simulator = Simulator(diagram)

context = simulator.get_mutable_context()

# Simulate
duration = 4.0
for i in range(5):
    context.SetTime(0.)
    context.SetContinuousState(np.random.randn(n_quadrotors*6 + n_balls*4,))
    simulator.Initialize()
    simulator.AdvanceTo(duration)