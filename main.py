import pydrake
import matplotlib.pyplot as plt
import numpy as np
import control 

from pydrake.all import DiagramBuilder, LinearQuadraticRegulator, Simulator, plot_system_graphviz 
from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder,
                         PlanarSceneGraphVisualizer, SceneGraph, TrajectorySource,
                         SnoptSolver, MultibodyPositionToGeometryPose, PiecewisePolynomial,
                         MathematicalProgram, JacobianWrtVariable, eq, le, ge, gt, lt, IpoptSolver, DirectCollocation, AddDirectCollocationConstraint, DirectCollocationConstraint,
                         InputPortIndex, Demultiplexer, Adder, Gain, FirstOrderTaylorApproximation)
from pydrake.autodiffutils import autoDiffToValueMatrix
from quadrotor2d import Quadrotor2D
from ball2d import Ball2D
from visualization import Visualizer
from ltv_controller import LTVController
from collisions import CalcClosestDistanceQuadBall, CalcPostCollisionStateQuadBall, CalcPostCollisionStateBallQuad, CalcPostCollisionStateQuadBallResidual, CalcPostCollisionStateBallQuadResidual

n_quadrotors = 1
n_balls = 1

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
            ltv_lqr = builder.AddSystem(LTVController(6,2))
            ltv_lqr.set_name('LTV LQR ' + str(i))

            builder.Connect(demulti_x.get_output_port(i), ltv_lqr.get_input_port(0))
            builder.Connect(quadrotor_plants[i].get_output_port(0), ltv_lqr.get_input_port(1))
            builder.Connect(demulti_u.get_output_port(i), ltv_lqr.get_input_port(2))
            builder.Connect(demulti_K.get_output_port(i), ltv_lqr.get_input_port(3))
            builder.Connect(ltv_lqr.get_output_port(0), quadrotor_plants[i].get_input_port(0))

        source_u = builder.AddSystem(TrajectorySource(trajectory_u))
        source_u.set_name('source feedforward input trajectory')
        source_x = builder.AddSystem(TrajectorySource(trajectory_x))
        source_x.set_name('source reference trajectory')
        demulti_source_x = builder.AddSystem(Demultiplexer([6*n_quadrotors, 4*n_balls]))
        demulti_source_x.set_name('quad and ball trajectories')

        source_K = builder.AddSystem(TrajectorySource(trajectory_K))
        source_K.set_name('source time-varying K')

        builder.Connect(source_u.get_output_port(0), demulti_u.get_input_port(0))
        builder.Connect(source_x.get_output_port(0), demulti_source_x.get_input_port(0))
        builder.Connect(demulti_source_x.get_output_port(0), demulti_x.get_input_port(0))
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
t_impact = 100 # time step where impact occurs
h_min = 0.005/2
h_max = 0.02/2

prog = MathematicalProgram()
h = prog.NewContinuousVariables(T, name='h')
u = prog.NewContinuousVariables(rows=T+1, cols = 2*n_quadrotors, name = 'u')
x = prog.NewContinuousVariables(rows=T+1, cols= 6*n_quadrotors+4*n_balls, name='q')
dv = prog.decision_variables()

prog.AddBoundingBoxConstraint([h_min] * T, [h_max] * T, h)

dir_coll_constr = DirectCollocationConstraint(diagram, context)

for t in range(T):
    if t == t_impact: # Don't add Direct collocation constraint at impact
        continue 
    AddDirectCollocationConstraint(dir_coll_constr, np.array([[h[t]]]), x[t,:].reshape(-1,1), x[t+1,:].reshape(-1,1), u[t,:].reshape(-1,1), u[t+1,:].reshape(-1,1), prog)

state_init = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
state_final = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0])

# Initial conditions
prog.AddLinearConstraint(eq(x[0,:], state_init))

# Final conditions
# prog.AddLinearConstraint(eq(x[T,0:6], state_final[0:6]))
# prog.AddLinearConstraint(eq(q[T,6:8], state_final[6:8]))

# Input constraints
for i in range(n_quadrotors):
    prog.AddLinearConstraint(ge(u[:,2*i],-50.0))
    prog.AddLinearConstraint(le(u[:,2*i], 50.0))
    prog.AddLinearConstraint(ge(u[:,2*i+1],-50.0))
    prog.AddLinearConstraint(le(u[:,2*i+1], 50.0))

# Don't allow quadrotor to pitch more than 60 degrees
for i in range(n_quadrotors):
    prog.AddLinearConstraint(ge(x[:,6*i+2],-np.pi/3))
    prog.AddLinearConstraint(le(x[:,6*i+2],np.pi/3))

# NOTE: If I comment out all of these impact constraints, the simulation runs smoothly. But with them, the simulation 
# freezes (at least on my computer.) The optimization solves fine, but the simulation doesn't work
# Impact constraint
quad_temp = Quadrotor2D()
for t in range(T):
    if t == t_impact:
        # At impact, witness function == 0
        prog.AddConstraint(lambda a: np.array([CalcClosestDistanceQuadBall(a[0:3], a[6:8])]).reshape(1,1), lb=np.zeros((1,1)), ub=np.zeros((1,1)), vars=x[t,:].reshape(-1,1))
        # At impact, enforce discrete collision update for both ball and quadrotor
        prog.AddConstraint(CalcPostCollisionStateQuadBallResidual, lb=np.zeros((6,1)), ub=np.zeros((6,1)), vars=np.concatenate((x[t,:], x[t+1, 0:6])).reshape(-1,1))
        prog.AddConstraint(CalcPostCollisionStateBallQuadResidual, lb=np.zeros((4,1)), ub=np.zeros((4,1)), vars=np.concatenate((x[t,:], x[t+1, 6:10])).reshape(-1,1))
    else:
        # Everywhere else, witness function must be > 0
        prog.AddConstraint(lambda a: np.array([CalcClosestDistanceQuadBall(a[0:3], a[6:8])]).reshape(1,1), lb=np.zeros((1,1)), ub=np.inf*np.ones((1,1)), vars=x[t,:].reshape(-1,1))


# Don't allow quadrotor collisions
# for i in range(n_quadrotors):
#     for j in range(i+1, n_quadrotors):
#         prog.AddConstraintToAllKnotPoints((prog.state()[6*i]-prog.state()[6*j])**2 +
#                                         (prog.state()[6*i+1]-prog.state()[6*j+1])**2 >= 0.65**2)


###############################################################################
# Set up initial guesses
quad_plant = Quadrotor2D()
h_init = h_max
x_initial = PiecewisePolynomial.FirstOrderHold([0, T * h_init], np.column_stack((state_init, state_final)))
u_initial = PiecewisePolynomial.ZeroOrderHold([0, T * h_init], 0.5*quad_plant.mass*quad_plant.gravity*np.ones((2*n_quadrotors,2)))
# NOTE: Right now not giving the optimizer an initial guess. Should probably add that in
# prog.SetInitialTrajectory(u_initial, x_initial)

# initial_guess = np.empty(prog.num_vars())
# # initial guess for the time step
# h_guess = h_max
# prog.SetDecisionVariableValueInVector(h, [h_guess] * T, initial_guess)
# # linear interpolation of the configuration
# q0_guess = np.array([0, -1.0, 0])
# q_guess_poly = PiecewisePolynomial.FirstOrderHold(
#     [0, T * h_guess],
#     np.column_stack((q0_guess, - q0_guess))
# )
# qd_guess_poly = q_guess_poly.derivative()
# qdd_guess_poly = q_guess_poly.derivative()
# # set initial guess for configuration, velocity, and acceleration
# q_guess = np.hstack([q_guess_poly.value(t * h_guess) for t in range(T + 1)]).T
# qd_guess = np.hstack([qd_guess_poly.value(t * h_guess) for t in range(T + 1)]).T
# qdd_guess = np.hstack([qdd_guess_poly.value(t * h_guess) for t in range(T)]).T

# prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
# prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
# prog.SetDecisionVariableValueInVector(qdd, qdd_guess, initial_guess)


solver = SnoptSolver()
print("Solving...")
result = solver.Solve(prog)
# be sure that the solution is optimal
assert result.is_success()

print(f'Solution found? {result.is_success()}.')

#################################################################################
# Extract results
# get optimal solution
h_opt = result.GetSolution(h)
x_opt = result.GetSolution(x)
u_opt = result.GetSolution(u)
time_breaks_opt = np.array([sum(h_opt[:t]) for t in range(T+1)])
u_opt_poly = PiecewisePolynomial.FirstOrderHold(time_breaks_opt, u_opt.T)
x_opt_poly = PiecewisePolynomial.Cubic(time_breaks_opt, x_opt.T, False)

#################################################################################
# Create list of K matrices for time varying LQR
context = quad_plant.CreateDefaultContext()
breaks = np.linspace(0, x_opt_poly.end_time(), 100)

K_samples = np.zeros((breaks.size, 12*n_quadrotors))

for i in range(n_quadrotors):
    K = None
    for j in range(breaks.size):
        context.SetContinuousState(x_opt_poly.value(breaks[j])[6*i:6*(i+1)])
        context.FixInputPort(0,u_opt_poly.value(breaks[j])[2*i:2*(i+1)])
        linear_system = FirstOrderTaylorApproximation(quad_plant,context)
        A = linear_system.A()
        B = linear_system.B()
        try:
            K, _, _ = control.lqr(A, B, Q, R)
        except:
            assert K is not None, "Failed to calculate initial K for quadrotor " + str(i)
            print ("Warning: Failed to calculate K at timestep", j, "for quadrotor", i, ". Using K from previous timestep")

        K_samples[j, 12*i:12*(i+1)] = K.reshape(-1)

K_poly = PiecewisePolynomial.ZeroOrderHold(breaks, K_samples.T)

##################################################################################
# Setup diagram for simulation
diagram = makeDiagram(n_quadrotors, n_balls, use_visualizer=True, trajectory_u = u_opt_poly, trajectory_x=x_opt_poly, trajectory_K = K_poly)

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
context.SetAccuracy(1e-4)

##############################################3
# # Simulate
duration = x_opt_poly.end_time()
# context.SetTime(0.)
# context.SetContinuousState(state_init)
# simulator.Initialize()
# simulator.AdvanceTo(duration)

t_arr = np.linspace(0,duration,100)
# context.SetTime(0.)
# context.SetContinuousState(state_init)
# simulator.Initialize()

# Plot
q_opt = np.zeros((100,6*n_quadrotors+4*n_balls))
q_actual = np.zeros((100,6*n_quadrotors+4*n_balls))
for i in range(100):
    t = t_arr[i]
    # simulator.AdvanceTo(t_arr[i])
    q_opt[i,:] = x_opt_poly.value(t).flatten()
    # q_actual[i,:] = context.get_continuous_state_vector().CopyToVector()

for i in range(n_quadrotors):
    ind_i = 6*i
    ind_f = ind_i + 3
    plt.figure(figsize=(6, 3))
    plt.plot(t_arr, q_opt[:,ind_i:ind_f])
    ind_i = 6*i + 3
    ind_f = ind_i + 3
    plt.figure(figsize=(6, 3))
    plt.plot(t_arr, q_opt[:,ind_i:ind_f])
    # plt.figure(figsize=(6, 3))
    # plt.plot(t_arr, q_actual[:,ind_i:ind_f])
    # plt.figure(figsize=(6, 3))
    # plt.plot(t_arr, q_actual[:,ind_i:ind_f]-q_opt[:,ind_i:ind_f])

for i in range(n_balls):
    ind_i = 6*n_quadrotors + 4*i
    ind_f = ind_i + 2
    plt.figure(figsize=(6, 3))
    plt.plot(t_arr, q_opt[:,ind_i:ind_f])
    ind_i = 6*n_quadrotors + 4*i + 2
    ind_f = ind_i + 2
    plt.figure(figsize=(6, 3))
    plt.plot(t_arr, q_opt[:,ind_i:ind_f])
    # plt.figure(figsize=(6, 3))
    # plt.plot(t_arr, q_actual[:,ind_i:ind_f])
    # plt.figure(figsize=(6, 3))
    # plt.plot(t_arr, q_actual[:,ind_i:ind_f]-q_opt[:,ind_i:ind_f])

dist = []
for t in range(t_arr.size):
    dist.append(CalcClosestDistanceQuadBall(q_opt[t, 0:3], q_opt[t, 6:8]))
plt.figure(figsize=(6, 3))
plt.plot(t_arr, dist)

if n_quadrotors >= 2:
    plt.figure()
    plt.figure(figsize=(6, 3))
    plt.plot(t_arr, np.linalg.norm(q_actual[:,0:2] - q_actual[:,6:8], axis=1))
plt.show()