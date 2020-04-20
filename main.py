import pydrake
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import DiagramBuilder, LinearQuadraticRegulator, Simulator, plot_system_graphviz
from quadrotor2d import Quadrotor2D
from ball2d import Ball2D
from visualization import Visualizer

def QuadrotorLQR(plant, n_quadrotors, n_balls):
    context = plant.CreateDefaultContext()
    context.SetContinuousState(np.zeros([6, 1]))
    context.FixInputPort(0, plant.mass * plant.gravity / 2. * np.array([1, 1]))
    for i in range(n_quadrotors):
        context.FixInputPort(1 + i, np.ones(6))
    for i in range(n_balls):
        context.FixInputPort(1 + n_quadrotors + i, np.ones(4))

    Q = np.diag([10, 10, 10, 1, 1, (plant.length / 2. / np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]])

    return LinearQuadraticRegulator(plant, context, Q, R)
# 
n_quadrotors = 1
n_balls = 1

builder = DiagramBuilder()
# Setup quadrotor plants and controllers
quadrotor_plants = []
quadrotor_controllers = []
for i in range(n_quadrotors):
    new_quad = Quadrotor2D(n_quadrotors=n_quadrotors-1, n_balls=n_balls)
    new_quad.set_name('quad_' + str(i))
    plant = builder.AddSystem(new_quad)
    quadrotor_plants.append(plant)
    controller = QuadrotorLQR(plant, n_quadrotors-1, n_balls)
    quadrotor_controllers.append(builder.AddSystem(controller))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

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

# # Setup visualization
# visualizer = builder.AddSystem(Visualizer(n_quadrotors=n_quadrotors, n_balls=n_balls))
# for i in range(n_quadrotors):
#     builder.Connect(quadrotor_plants[i].get_output_port(0), visualizer.get_input_port(i))
# for i in range(n_balls):
#     builder.Connect(ball_plants[i].get_output_port(0), visualizer.get_input_port(n_quadrotors + i))



diagram = builder.Build()
diagram.ToAutoDiffXd()
context = diagram.CreateDefaultContext()
subsystem = diagram.GetSubsystemByName('quad_0')
sub_context = diagram.GetMutableSubsystemContext(subsystem,context)
# sub_context.SetContinuousState(np.array([0.0, -1.5 , 0.0, 0.]))
# sub_context.FixInputPort(0,np.array([0.0,0.0]))
# print(sub_context.ssxt))
print(subsystem.EvalTimeDerivatives(sub_context))
# EvalTimeDerivatives

def TemplateResidualFunction(vars_quads, vars_balls):

    context = diagram.CreateDefaultContext()
    split_quad = [3,6,9]
    split_ball = [2,4]

    for i in range(n_quadrotors):
        sub_system = diagram.GetSubsystemByName('quad_' + str(i))
        sub_context = diagram.GetMutableSubsystemContext(sub_system,context)
        q, qd, qdd, f = np.split(vars_quads[i], split_at)
        sub_context.SetContinuousState(np.concatenate((q, qd)))
        sub_context.FixInputPort(0, f)

    for i in range(n_balls):
        sub_system = diagram.GetSubsystemByName('ball_' + str(i))
        sub_context = diagram.GetMutableSubsystemContext(sub_system,context)
        q, qd, qdd = np.split(vars_balls[i], split_ball)
        sub_context.SetContinuousState(np.concatenate((q, qd)))

    diagram.EvalTimeDerivatives(context).get_vector()


    # nq = 3*n_quadrotors + 2*n_balls
    # split_at = [nq,2*nq,3*nq]
    # q, qd, qdd, f = np.split(vars, split_at)

    # continuous_state = np.array([])

    # for i in range(n_quadrotors):
    #     continuous_state = np.concatenate((continuous_state, q[3*i:3*(i+1)]))
    #     continuous_state = np.concatenate((continuous_state,qd[3*i:3*(i+1)]))
    
    # for j in range(n_balls):
    #     continuous_state = np.concatenate((continuous_state, q[3*(i+1)+2*i:3*(i+1)+2*(i+1)]))
    #     continuous_state = np.concatenate((continuous_state,qd[3*(i+1)+2*i:3*(i+1)+2*(i+1)]))




# plt.figure(figsize=(20, 10))
# plot_system_graphviz(diagram)
# plt.show()
# Set up a simulator to run this diagram
simulator = Simulator(diagram)
integrator = simulator.get_mutable_integrator()
integrator.set_maximum_step_size(0.01) # Reduce the max step size so that we can always detect collisions
context = simulator.get_mutable_context()

##############################################3
# state_init = np.array([-1.,1.,1.,0.,1.,1.,-1.,0.])
# state_init = np.random.randn(n_quadrotors*6 + n_balls*4,)
# state_init = np.array([0.0 ,1.,0.,0., 0.0 ,1.3,0.,0.,0.,1.8,0.0,-0.8])
# state_init = np.array([0.0, 0.0 , 0., 0., 0., 0.,  -1.0, 1.0, 0.8, -0.8])
state_init = np.array([0.0, -1.0 , 0.0, 0., 0., 0.,  -0.2, 1.0, 0.0, -10.0])
# state_init = np.array([0.0, -0.3 , 0.0, 0., 0., 0.,0.0, 0.3 , 0.0, 0., 0., 0.,  0.0, 0.0, 0.0, 0.0])

# Simulate
duration = 4.0
for i in range(5):
    context.SetTime(0.)
    context.SetContinuousState(state_init)
    simulator.Initialize()
    simulator.AdvanceTo(duration)