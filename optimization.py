from pydrake.all import eq, MathematicalProgram, Solve, Variable
import numpy as np

class QuadController(VectorSystem):
    
    def __init__(self, quad_plants, ball_plants):
        
        # n_x inputs: 6 * n_quad + 4 * n_balls
        # n_u outputs: spring preload
    	self.n_u = 2*len(quad_plants)
		self.n_x = 6*len(quad_plants) + 4*len(ball_plants)
		self.n_quads = len(quad_plants)
		self.n_balls = len(ball_plants)
        VectorSystem.__init__(self, self.n_x,  self.n_u)
        self.quad_plants = quad_plants
        self.ball_plants = ball_plants
        # input controller
        self.u = u = np.empty((n_u, 1), dtype=Variable)
        
        self.last_state = np.full(self.n_x, np.nan)
    
    # note that this function is called at each time step
    def DoCalcVectorOutput(self, context, state, unused, delta):
        
        # update controller internal memory
        self.last_state = state
        
    def dynamic_constraint_quad(self, state):

    	return residuals
    def dynamic_constraint_ball(self, state):

    	return residuals

	def optimization(self, quad_plants, ball_plants, contexts):
		prog = MathematicalProgram()
		#Time steps likely requires a line search. 
		#Find the number of time steps until a ball reaches a threshold so that a quadrotor can catch it.
		N = 200 #Number of time steps
		n_u = 2*len(plants)
		n_x = 6*len(quad_plants) + 4*len(ball_plants)
		n_quads = len(quad_plants)
		n_balls = len(ball_plants)
		u = np.empty((self.n_u, N-1), dtype=Variable)
		x = np.empty((self.n_x, N), dtype=Variable)
		I =  0.00383
		r = 0.25
		mass = 0.486
		g = 9.81
		#Add all the decision variables
		for n in range(N-1):
			u[:,n] = prog.NewContinuousVariables(self.n_u, 'u' + str(n))
			x[:,n] = prog.NewContinuousVariables(self.n_x, 'x' + str(n))
		x[:,N-1] = prog.NewContinuousVariables(self.n_x, 'x' + str(N))

		#Start at the correct initial conditions
		#Connect quadrotor and ball state variables to the initial condition of mathprog
		x0 = np.empty((self.n_x,1))
		for k in range(n_quads + n_balls):
			if k < n_quads:
				x0[6*k:6*k+5] = self.quad_plants[k] #???
		prog.AddBoundingBoxConstraint(x0, x0, x[:,0])
		for n in range(N-1):
			for k in range(n_quads + n_balls):
				#Handles no collisions
				if k < n_quads:
					#Quad dynamic constraints
					dynamics_constraint_vel = #?
					dynamics_constraint_acc = #?
					prog.AddConstraint(dynamics_constraint_vel[0, 0])
	  				prog.AddConstraint(dynamics_constraint_vel[1, 0])
					prog.AddConstraint(dynamics_constraint_vel[2, 0])
					prog.AddConstraint(dynamics_constraint_acc[0, 0])
	  				prog.AddConstraint(dynamics_constraint_acc[1, 0])
					prog.AddConstraint(dynamics_constraint_acc[2, 0])
				else:
					#Ball dynamic constraints
					dynamics_constraint_vel = #
					dynamics_constraint_acc = #
					prog.AddConstraint(dynamics_constraint_vel[0, 0])
					prog.AddConstraint(dynamics_constraint_vel[1, 0])
					prog.AddConstraint(dynamics_constraint_acc[0, 0])
	  				prog.AddConstraint(dynamics_constraint_acc[1, 0])

	  	#Final conditions they end in the same location

