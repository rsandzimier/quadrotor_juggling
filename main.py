import pydrake
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import DiagramBuilder, LinearQuadraticRegulator, Simulator, plot_system_graphviz 
from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder,
                         PlanarSceneGraphVisualizer, SceneGraph, TrajectorySource,
                         SnoptSolver, MultibodyPositionToGeometryPose, PiecewisePolynomial,
                         MathematicalProgram, JacobianWrtVariable, eq, le, ge,IpoptSolver)
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
diagram = diagram.ToAutoDiffXd()

def pack_vars(vars_quads, vars_balls):
    vars = np.zeros(0)
    for i in range(n_quadrotors):
        vars = np.append(vars, vars_quads[i])
    for i in range(n_balls):
        vars = np.append(vars, vars_balls[i])
    return vars

def unpack_vars(vars):
    nvars_quads = 11 # q: 3, qd: 3, qdd: 3, u: 2
    nvars_balls = 6 # q: 2, qd: 2, qdd: 2
    vars_quads = []
    vars_balls = []
    for i in range(n_quadrotors):
        ind = nvars_quads*i
        vars_quads.append(vars[ind:ind+nvars_quads])
    for i in range(n_balls):
        ind = nvars_quads*n_quadrotors + nvars_balls*i
        vars_balls.append(vars[ind:ind+nvars_balls])
    return vars_quads, vars_balls

def TemplateResidualFunction(vars):
    vars_quads, vars_balls = unpack_vars(vars)
    context = diagram.CreateDefaultContext()
    split_quad = [3,6,9]
    split_ball = [2,4]

    for i in range(n_quadrotors):
        sub_system = diagram.GetSubsystemByName('quad_' + str(i))
        sub_context = diagram.GetMutableSubsystemContext(sub_system,context)
        q, qd, qdd, f = np.split(vars_quads[i], split_quad)
        sub_context.SetContinuousState(np.concatenate((q, qd)))
        sub_context.FixInputPort(0, f)

    for i in range(n_balls):
        sub_system = diagram.GetSubsystemByName('ball_' + str(i))
        sub_context = diagram.GetMutableSubsystemContext(sub_system,context)
        q, qd, qdd = np.split(vars_balls[i], split_ball)
        sub_context.SetContinuousState(np.concatenate((q, qd)))

    return qdd - diagram.EvalTimeDerivatives(context).CopyToVector()[3:]

def create_context(vars):
    vars_quads, vars_balls = unpack_vars(vars)
    context = diagram.CreateDefaultContext()
    split_quad = [3,6,9]
    split_ball = [2,4]

    for i in range(n_quadrotors):
        sub_system = diagram.GetSubsystemByName('quad_' + str(i))
        sub_context = diagram.GetMutableSubsystemContext(sub_system,context)
        q, qd, qdd, f = np.split(vars_quads[i], split_quad)
        sub_context.SetContinuousState(np.concatenate((q, qd)))
        sub_context.FixInputPort(0, f)

    for i in range(n_balls):
        sub_system = diagram.GetSubsystemByName('ball_' + str(i))
        sub_context = diagram.GetMutableSubsystemContext(sub_system,context)
        q, qd, qdd = np.split(vars_balls[i], split_ball)
        sub_context.SetContinuousState(np.concatenate((q, qd)))

    return context

def CollResid(vars):
    
    vars_k = vars[:11]
    vars_kplus = vars[11:]

    context = diagram.CreateDefaultContext()
    split_quad = [3,6,9]
    split_ball = [2,4]

    #This is sketchily written and will only for one quad
    vars_quads_k, vars_balls_k = unpack_vars(vars_k)
    vars_quads_kplus, vars_balls_kplus = unpack_vars(vars_k)
    _,_,_,f_k     = np.split(vars_quads_k[0], split_quad)
    _,_,_,f_kplus = np.split(vars_quads_kplus[0], split_quad)

    context_k = create_context(vars_k)
    context_kplus  = create_context(vars_kplus)
    
    x_k = context_k.get_continuous_state_vector().CopyToVector()
    x_kplus = context_kplus.get_continuous_state_vector().CopyToVector()

    x_dot_k = diagram.EvalTimeDerivatives(context_k).CopyToVector()[:]
    x_dot_kplus = diagram.EvalTimeDerivatives(context_kplus).CopyToVector()[:]

    x_ck =  0.5*(x_k + x_kplus)   + 0.125*h*(x_dot_k - x_dot_kplus)
    x_dot_ck = (-1.5/h)*(x_k - x_kplus) - 0.25*(x_dot_k + x_dot_kplus)
    f_ck = 0.5*(f_k + f_kplus)

    context_c = diagram.CreateDefaultContext()
    context_c.SetContinuousState(x_ck)
    sub_system = diagram.GetSubsystemByName('quad_0')
    sub_context = diagram.GetMutableSubsystemContext(sub_system,context_c)
    sub_context.FixInputPort(0, f_ck)

    return x_dot_ck - diagram.EvalTimeDerivatives(context_c).CopyToVector()[:]

prog = MathematicalProgram()
# vector of the time intervals
T = 400#Number of breakpoints
# (distances between the T + 1 break points)

h = 0.02
# h = prog.NewContinuousVariables(T, name='h')
# h_min = 0.001
# h_max = 0.01
# prog.AddBoundingBoxConstraint([h_min] * T, [h_max] * T, h)

nq_quad = 3
n_u = 2
# system configuration, generalized velocities, and accelerations
u = prog.NewContinuousVariables(rows=T, cols = n_u, name = 'u')
q = prog.NewContinuousVariables(rows=T+1, cols=nq_quad, name='q')
qd = prog.NewContinuousVariables(rows=T+1, cols=nq_quad, name='qd')
qdd = prog.NewContinuousVariables(rows=T, cols=nq_quad, name='qdd')



# Implicit euler constraints for velocity and acceleration
# for t in range(T):
#     # prog.AddConstraint(eq(q[t+1], q[t] + h[t] * qd[t]))
#     # prog.AddConstraint(eq(qd[t+1], qd[t] + h[t] * qdd[t]))
#     prog.AddConstraint(eq(q[t+1], q[t] + h * qd[t]))
#     prog.AddConstraint(eq(qd[t+1], qd[t] + h * qdd[t]))

# # residual equations for all t using Implicit Euler
# for t in range(T):
#     vars_quads = [np.concatenate((q[t+1], qd[t+1], qdd[t], u[t]))]
#     vars_balls = []
#     vars = pack_vars(vars_quads, vars_balls)
#     nqdd = nq_quad*n_quadrotors
#     prog.AddConstraint(TemplateResidualFunction, lb=[0]*nqdd, ub=[0]*nqdd, vars = vars)

for t in range(T-1):
    vars_quads = [np.concatenate((q[t], qd[t], qdd[t], u[t]))]
    vars_balls = []
    vars_k = pack_vars(vars_quads, vars_balls)

    vars_quads = [np.concatenate((q[t+1], qd[t+1], qdd[t+1], u[t+1]))]
    vars_balls = []
    vars_kplus = pack_vars(vars_quads, vars_balls)

    vars = np.concatenate((vars_k,vars_kplus))
    nqdd = nq_quad*n_quadrotors*2
    prog.AddConstraint(CollResid, lb=[0]*nqdd, ub=[0]*nqdd, vars = vars)


# for t in range(T-1):
    # inputs to Coll_resid
    # u_ct = 0.5*(u[t] + u[t+1])
    # x_ct = 0.5*(q[t]+q[t+1])   + 0.125*h*(qd[t] - qd[t+1])
    # v_ct = 0.5*(qd[t]+qd[t+1]) + 0.125*h*(qdd[t] - qdd[t+1])
    
    # [np.concatenate((q[t+1], qd[t+1], qdd[t], u[t]))]
    # qd_ct = 
    # qdd_ct =

state_init = np.array([0.5, 0.0 , 0.0, 0., 0., 0.0])
state_final = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
split_quad = [3,6]

# Initial conditions
q0, qd0 = np.split(state_init, [3])

prog.AddLinearConstraint(eq(q[0], q0))
prog.AddLinearConstraint(eq(qd[0], qd0))

# Final conditions
qf, qdf = np.split(state_final, [3])
prog.AddLinearConstraint(eq(q[T], qf))
prog.AddLinearConstraint(eq(qd[T], qdf))

for t in range(T):
    prog.AddLinearConstraint(ge(u[t],np.zeros(2)))
    prog.AddLinearConstraint(le(u[t],10*np.ones(2)))
###############################################################################
# Set up initial guesses
# vector of the initial guess
initial_guess = np.empty(prog.num_vars())

# initial guess for the time step
# h_guess = h_max
# prog.SetDecisionVariableValueInVector(h, [h_guess] * T, initial_guess)

# linear interpolation of the configuration
q0_guess = np.array([0, -1.0, 0])
q_guess_poly = PiecewisePolynomial.FirstOrderHold(
    [0, T * h],
    np.column_stack((q0_guess, - q0_guess))
)
qd_guess_poly = q_guess_poly.derivative()
qdd_guess_poly = q_guess_poly.derivative()

# set initial guess for configuration, velocity, and acceleration
q_guess = np.hstack([q_guess_poly.value(t * h) for t in range(T + 1)]).T
qd_guess = np.hstack([qd_guess_poly.value(t * h) for t in range(T + 1)]).T
qdd_guess = np.hstack([qdd_guess_poly.value(t * h) for t in range(T)]).T
prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qd, qd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qdd, qdd_guess, initial_guess)
solver = SnoptSolver()
print("Solving...")
result = solver.Solve(prog)
# be sure that the solution is optimal
assert result.is_success()

print(f'Solution found? {result.is_success()}.')

#################################################################################
# Extract results
# get optimal solution
# h_opt = result.GetSolution(h)
q_opt = result.GetSolution(q)
qd_opt = result.GetSolution(qd)
qdd_opt = result.GetSolution(qdd)
u_opt = result.GetSolution(u)

##################################################################################
# Setup diagram for simulation
time_breaks_opt = np.array([t*h for t in range(T)])
print(time_breaks_opt[-1])
u_opt_poly = PiecewisePolynomial.ZeroOrderHold(time_breaks_opt, u_opt.T)
diagram = makeDiagram(n_quadrotors, n_balls, use_visualizer=True, trajectory=u_opt_poly)
# print(u_opt_poly)
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
# state_init = np.array([-1.,1.,1.,0.,1.,1.,-1.,0.])
# state_init = np.random.randn(n_quadrotors*6 + n_balls*4,)
# state_init = np.array([0.0 ,1.,0.,0., 0.0 ,1.3,0.,0.,0.,1.8,0.0,-0.8])
# state_init = np.array([0.0, 0.0 , 0., 0., 0., 0.,  -1.0, 1.0, 0.8, -0.8])
# state_init = np.array([0.0, -1.0 , 0.0, 0., 0., 0.,  -0.2, 1.0, 0.0, -10.0])
# state_init = np.array([0.0, -0.3 , 0.0, 0., 0., 0.,0.0, 0.3 , 0.0, 0., 0., 0.,  0.0, 0.0, 0.0, 0.0])
# state_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# # Simulate
duration = time_breaks_opt[-1]
for i in range(5):
    context.SetTime(0.)
    context.SetContinuousState(state_init)
    simulator.Initialize()
    simulator.AdvanceTo(duration)