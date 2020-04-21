import pydrake
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import DiagramBuilder, LinearQuadraticRegulator, Simulator, plot_system_graphviz 
from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder,
                         PlanarSceneGraphVisualizer, SceneGraph, TrajectorySource,
                         SnoptSolver, MultibodyPositionToGeometryPose, PiecewisePolynomial,
                         MathematicalProgram, JacobianWrtVariable, eq)
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
n_balls = 0

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
#print(subsystem.EvalTimeDerivatives(sub_context))
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

def manipulator_equations(vars):

    context = diagram.CreateDefaultContext()
    split_quad = [3,6,9]
    split_ball = [2,4]

    sub_system = diagram.GetSubsystemByName('quad_0')
    sub_context = diagram.GetMutableSubsystemContext(sub_system,context)
    q, qd, qdd, u = np.split(vars, split_quad)
    q_list = []
    qd_list = []
    for i in range(len(q)):
        q_list.append(q[i].value())
        qd_list.append(qd[i].value())

    q_arr = np.array(q_list)
    qd_arr = np.array(qd_list)
    sub_context.SetContinuousState(np.concatenate((q_arr, qd_arr)))
    M = sub_system.CalcMassMatrix(sub_context)
    G = sub_system.CalcGravTerm(sub_context)
    InputVec = sub_system.CalcInputVec(sub_context, u)
    # return violation of the manipulator equations
    return M.dot(qdd) - InputVec - G


sub_system = diagram.GetSubsystemByName('quad_' + str(i))
sub_context = diagram.GetMutableSubsystemContext(sub_system,context)

prog = MathematicalProgram()
# vector of the time intervals
T = 200#Number of breakpoints
# (distances between the T + 1 break points)
h = prog.NewContinuousVariables(T, name='h')
nq_quad = 3
n_u = 2
# system configuration, generalized velocities, and accelerations
u = prog.NewContinuousVariables(rows=T, cols = n_u, name = 'u')
q = prog.NewContinuousVariables(rows=T+1, cols=nq_quad, name='q')
qd = prog.NewContinuousVariables(rows=T+1, cols=nq_quad, name='qd')
qdd = prog.NewContinuousVariables(rows=T, cols=nq_quad, name='qdd')
h_min = 0.01
h_max = 0.1
prog.AddBoundingBoxConstraint([h_min] * T, [h_max] * T, h)

# Implicit euler constraints for velocity and acceleration
for t in range(T):
    prog.AddConstraint(eq(q[t+1], q[t] + h[t] * qd[t+1]))
    prog.AddConstraint(eq(qd[t+1], qd[t] + h[t] * qdd[t]))

# Manipulator residual equations for all t using Implicit Euler
for t in range(T):
    vars = np.concatenate((q[t+1], qd[t+1], qdd[t], u[t]))
    prog.AddConstraint(manipulator_equations, lb=[0]*nq_quad, ub=[0]*nq_quad, vars =  vars)

state_init = np.array([0.0, -1.0 , 0.0, 0., 0., 0.0])
state_final = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
split_quad = [3,6]

# Initial conditions
q0, qd0 = np.split(state_init, [3])

prog.AddLinearConstraint(eq(q[0], q0))
prog.AddLinearConstraint(eq(qd[0], qd0))

# Final conditions
qf, qdf= np.split(state_final, [3])
prog.AddLinearConstraint(eq(q[T], qf))
prog.AddLinearConstraint(eq(qd[T], qdf))


###############################################################################
# Set up initial guesses
# vector of the initial guess
initial_guess = np.empty(prog.num_vars())

# initial guess for the time step
h_guess = h_max
prog.SetDecisionVariableValueInVector(h, [h_guess] * T, initial_guess)

# linear interpolation of the configuration
q0_guess = np.array([0, -1.0, 0])
q_guess_poly = PiecewisePolynomial.FirstOrderHold(
    [0, T * h_guess],
    np.column_stack((q0_guess, - q0_guess))
)
qd_guess_poly = q_guess_poly.derivative()
qdd_guess_poly = q_guess_poly.derivative()

# set initial guess for configuration, velocity, and acceleration
q_guess = np.hstack([q_guess_poly.value(t * h_guess) for t in range(T + 1)]).T
qd_guess = np.hstack([qd_guess_poly.value(t * h_guess) for t in range(T + 1)]).T
qdd_guess = np.hstack([qdd_guess_poly.value(t * h_guess) for t in range(T)]).T
prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qdd, qdd_guess, initial_guess)
solver = SnoptSolver()
result = solver.Solve(prog)

# be sure that the solution is optimal
assert result.is_success()

print(f'Solution found? {result.is_success()}.')

#################################################################################
# Extract results
# get optimal solution
h_opt = result.GetSolution(h)
q_opt = result.GetSolution(q)
qd_opt = result.GetSolution(qd)
qdd_opt = result.GetSolution(qdd)
u_opt = result.GetSolution(u)

##################################################################################
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
# state_init = np.array([-1.,1.,1.,0.,1.,1.,-1.,0.])
# state_init = np.random.randn(n_quadrotors*6 + n_balls*4,)
# state_init = np.array([0.0 ,1.,0.,0., 0.0 ,1.3,0.,0.,0.,1.8,0.0,-0.8])
# state_init = np.array([0.0, 0.0 , 0., 0., 0., 0.,  -1.0, 1.0, 0.8, -0.8])
# state_init = np.array([0.0, -1.0 , 0.0, 0., 0., 0.,  -0.2, 1.0, 0.0, -10.0])
# state_init = np.array([0.0, -0.3 , 0.0, 0., 0., 0.,0.0, 0.3 , 0.0, 0., 0., 0.,  0.0, 0.0, 0.0, 0.0])

# # Simulate
# duration = 4.0
# for i in range(5):
#     context.SetTime(0.)
#     context.SetContinuousState(state_init)
#     simulator.Initialize()
#     simulator.AdvanceTo(duration)