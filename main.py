import pydrake
import matplotlib.pyplot as plt
import numpy as np
import control 

from pydrake.all import DiagramBuilder, LinearQuadraticRegulator, Simulator, plot_system_graphviz 
from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder,
                         PlanarSceneGraphVisualizer, SceneGraph, TrajectorySource,
                         SnoptSolver, MultibodyPositionToGeometryPose, PiecewisePolynomial,
                         MathematicalProgram, JacobianWrtVariable, eq, le, ge,IpoptSolver, DirectCollocation,
                         InputPortIndex, Demultiplexer, Adder, Gain, FirstOrderTaylorApproximation)
from pydrake.autodiffutils import autoDiffToValueMatrix
from quadrotor2d import Quadrotor2D
from ball2d import Ball2D
from visualization import Visualizer
from matrix_multiplier import MatrixMultiplier

n_quadrotors = 2
n_balls = 0

Q = np.diag([100000, 100000, 100000, 10000, 10000, 10000*(0.25 / 2. / np.pi)])
R = np.array([[0.1, 0.05], [0.05, 0.1]])

def QuadrotorLQR(plant):
    context = plant.CreateDefaultContext()
    context.SetContinuousState(np.zeros([6, 1]))
    context.SetContinuousState(np.array([0.0, 0.0,  0.0, 0.0, 0.0, 0.0]))
    context.FixInputPort(0, plant.mass * plant.gravity / 2. * np.array([1, 1]))

    Q = np.diag([100000, 100000, 100000, 10000, 10000, 10000*(plant.length / 2. / np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]])

    return LinearQuadraticRegulator(plant, context, Q, R)

def makeDiagram(n_quadrotors, n_balls, use_visualizer=False,trajectory_u=None, trajectory_x=None, trajectory_K = None):
    builder = DiagramBuilder()
    # Setup quadrotor plants and controllers
    quadrotor_plants = []
    quadrotor_controllers = []
    for i in range(n_quadrotors):
        new_quad = Quadrotor2D(n_quadrotors=n_quadrotors-1, n_balls=n_balls)
        new_quad.set_name('quad_' + str(i))
        plant = builder.AddSystem(new_quad)
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
        visualizer.set_name('visualizer')
        for i in range(n_quadrotors):
            builder.Connect(quadrotor_plants[i].get_output_port(0), visualizer.get_input_port(i))
        for i in range(n_balls):
            builder.Connect(ball_plants[i].get_output_port(0), visualizer.get_input_port(n_quadrotors + i))

    # Setup trajectory source


    if trajectory_x is not None and trajectory_u is not None and trajectory_K is not None:
        demulti_u = builder.AddSystem(Demultiplexer(2*n_quadrotors, 2))
        demulti_u.set_name('feedforward input')
        demulti_x = builder.AddSystem(Demultiplexer(6*n_quadrotors, 6))
        demulti_x.set_name('reference trajectory')
        demulti_K = builder.AddSystem(Demultiplexer(12*n_quadrotors, 12))
        demulti_K.set_name('time-varying K')

        for i in range(n_quadrotors):
            adder_ia = builder.AddSystem(Adder(2,2))
            adder_ia.set_name('sum control signals'+str(i))

            adder_ib = builder.AddSystem(Adder(2,6))
            adder_ib.set_name('quad_'+str(i)+' error')
            gain_ia = builder.AddSystem(Gain(-1,6))
            gain_ia.set_name('-1 gain_error'+str(i))

            matrix_multiplier = builder.AddSystem(MatrixMultiplier((2,6),(6,1)))
            matrix_multiplier.set_name('mat_mult_'+str(i))

            gain_ib = builder.AddSystem(Gain(-1,2))
            gain_ib.set_name('-1 gain_u'+str(i))

            builder.Connect(demulti_u.get_output_port(i), adder_ia.get_input_port(0))
            builder.Connect(gain_ib.get_output_port(0), adder_ia.get_input_port(1))
            builder.Connect(adder_ia.get_output_port(0), quadrotor_plants[i].get_input_port(0))
            builder.Connect(demulti_x.get_output_port(i),gain_ia.get_input_port(0))
            builder.Connect(gain_ia.get_output_port(0),adder_ib.get_input_port(0))
            builder.Connect(quadrotor_plants[i].get_output_port(0),adder_ib.get_input_port(1))
            builder.Connect(demulti_K.get_output_port(i),matrix_multiplier.get_input_port(0))
            builder.Connect(adder_ib.get_output_port(0),matrix_multiplier.get_input_port(1))
            builder.Connect(matrix_multiplier.get_output_port(0),gain_ib.get_input_port(0))

        source_u = builder.AddSystem(TrajectorySource(trajectory_u))
        source_u.set_name('source feedforward input trajectory')
        source_x = builder.AddSystem(TrajectorySource(trajectory_x))
        source_x.set_name('source reference trajectory')
        source_K = builder.AddSystem(TrajectorySource(trajectory_K))
        source_K.set_name('source time-varying K')

        builder.Connect(source_u.get_output_port(0), demulti_u.get_input_port(0))
        builder.Connect(source_x.get_output_port(0), demulti_x.get_input_port(0))
        builder.Connect(source_K.get_output_port(0), demulti_K.get_input_port(0))

    else:
        demulti_u = builder.AddSystem(Demultiplexer(2*n_quadrotors, 2))
        demulti_u.set_name('quadrotor input')
        for i in range(n_quadrotors):
            builder.Connect(demulti_u.get_output_port(i), quadrotor_plants[i].get_input_port(0))

        builder.ExportInput(demulti_u.get_input_port(0))
        
    diagram = builder.Build()
    return diagram

diagram = makeDiagram(n_quadrotors, n_balls, use_visualizer=False)
# plt.figure(figsize=(20, 10))
# plot_system_graphviz(diagram)
# plt.show()
###
# exit()
context = diagram.CreateDefaultContext()
T = 200 #Number of breakpoints
h_min = 0.005
h_max = 0.02
prog = DirectCollocation(diagram, context, T, h_min, h_max)

state_init = np.array([-1.0, 0.0 , 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 , 0.0, 0.0, 0.0, 0.0])
state_final = np.array([1.0, 0.0 , 0.0, 0.0, 0.0, 0.0, -1.0, 0.0 , 0.0, 0.0, 0.0, 0.0])
# state_init = np.array([-1.0, 0.0 , 0.0, 0.0, 0.0, 0.0])
# state_final = np.array([1.0, 0.0 , 0.0, 0.0, 0.0, 0.0])

# Initial conditions
prog.AddLinearConstraint(eq(prog.initial_state(), state_init))

# Final conditions
prog.AddLinearConstraint(eq(prog.final_state(), state_final))

# Input constraints
for i in range(n_quadrotors):
    prog.AddConstraintToAllKnotPoints(prog.input()[2*i]>=0.0)
    prog.AddConstraintToAllKnotPoints(prog.input()[2*i]<=10.0)
    prog.AddConstraintToAllKnotPoints(prog.input()[2*i+1]>=0.0)
    prog.AddConstraintToAllKnotPoints(prog.input()[2*i+1]<=10.0)

# Don't allow quadrotor collisions
for i in range(n_quadrotors):
    for j in range(i+1, n_quadrotors):
        prog.AddConstraintToAllKnotPoints((prog.state()[6*i]-prog.state()[6*j])**2 +
                                        (prog.state()[6*i+1]-prog.state()[6*j+1])**2 >= 0.65**2)


###############################################################################
# Set up initial guesses
quad_plant = Quadrotor2D()
h_init = h_max
x_initial = PiecewisePolynomial.FirstOrderHold([0, T * h_init], np.column_stack((state_init, state_final)))
u_initial = PiecewisePolynomial.ZeroOrderHold([0, T * h_init], 0.5*quad_plant.mass*quad_plant.gravity*np.ones((2*n_quadrotors,2)))
prog.SetInitialTrajectory(u_initial, x_initial)

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

#################################################################################
# Create list of K matrices for time varying LQR
context = quad_plant.CreateDefaultContext()
breaks = np.linspace(0, x_opt_poly.end_time(), 100)

K_samples = np.zeros((breaks.size, 12*n_quadrotors))

for i in range(n_quadrotors):
    for j in range(breaks.size):
        context.SetContinuousState(x_opt_poly.value(breaks[j])[6*i:6*(i+1)])
        context.FixInputPort(0,u_opt_poly.value(breaks[j])[2*i:2*(i+1)])
        linear_system = FirstOrderTaylorApproximation(quad_plant,context)
        A = linear_system.A()
        B = linear_system.B()
        K, _, _ = control.lqr(A, B, Q, R)
        K_samples[j, 12*i:12*(i+1)] = K.reshape(-1)

K_poly = PiecewisePolynomial.ZeroOrderHold(breaks, K_samples.T)

##################################################################################
# Setup diagram for simulation
diagram = makeDiagram(n_quadrotors, n_balls, use_visualizer=True, trajectory_u = u_opt_poly, trajectory_x=x_opt_poly, trajectory_K = K_poly)

###################################################################################
# Animate
plt.figure(figsize=(20, 10))
plot_system_graphviz(diagram)
plt.show()
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
q_opt = np.zeros((100,6*n_quadrotors))
q_actual = np.zeros((100,6*n_quadrotors))
for i in range(100):
    t = t_arr[i]
    simulator.AdvanceTo(t_arr[i])
    q_opt[i,:] = x_opt_poly.value(t).flatten()
    q_actual[i,:] = context.get_continuous_state_vector().CopyToVector()

for i in range(n_quadrotors):
    ind_i = 6*i
    ind_f = ind_i + 3
    plt.figure()
    plt.plot(t_arr, q_opt[:,ind_i:ind_f])
    plt.plot(t_arr, q_actual[:,ind_i:ind_f])
    plt.figure()
    plt.plot(t_arr, q_actual[:,ind_i:ind_f]-q_opt[:,ind_i:ind_f])

if n_quadrotors >= 2:
    plt.figure()
    plt.plot(t_arr, np.linalg.norm(q_actual[:,0:2] - q_actual[:,6:8], axis=1))
plt.show()