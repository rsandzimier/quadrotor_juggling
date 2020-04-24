import numpy as np

from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType, WitnessFunctionDirection, UnrestrictedUpdateEvent_
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.symbolic import Expression
from collisions import CalcClosestDistanceQuadBall, CalcPostCollisionStateQuadBall

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html

# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
@TemplateSystem.define("Quadrotor2D_")
def Quadrotor2D_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, n_quadrotors=0, n_balls=0, converter=None):
            # n_quadrotors is the number of OTHER quadrotors (besides this one)
            LeafSystem_[T].__init__(self, converter)
            self.n_quadrotors = n_quadrotors
            self.n_balls = n_balls
            # two inputs (thrust)
            self.DeclareVectorInputPort("u", BasicVector_[T](2))

            # six outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](6),
                                         self.CopyStateOut)
            # three positions, three velocities
            self.DeclareContinuousState(3, 3, 0)

            # Other quadrotor states as inputs
            for i in range(self.n_quadrotors):
                    self.DeclareVectorInputPort("quad_" + str(i), BasicVector_[T](6))

            # Ball states as inputs
            for i in range(self.n_balls):
                self.DeclareVectorInputPort("ball_" + str(i), BasicVector_[T](4))

            self.witness_functions = []
            witness_callback_ball = lambda ball_ind: lambda context: self.WitnessCollisionBall(context, ball_ind)
            event_callback_ball = lambda ball_ind: lambda context, event, state: self.EventCollisionBall(context, event, state, ball_ind)
            for i in range(self.n_balls):
                witness_function = self.MakeWitnessFunction("collision_ball_" + str(i), WitnessFunctionDirection.kPositiveThenNonPositive,
                            witness_callback_ball(i), UnrestrictedUpdateEvent_[T](event_callback_ball(i)))
                self.witness_functions.append(witness_function)

            # parameters based on [Bouadi, Bouchoucha, Tadjine, 2007]
            self.length = 0.25  # length of rotor arm
            self.contact_length = 0.3 # half-width of base
            self.height = 0.025 # half-height of base
            self.mass = 0.486  # mass of quadrotor
            self.inertia = 0.00383  # moment of inertia
            self.gravity = 9.81  # gravity
            self.stiffness_quad = 10000.0

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, n_quadrotors=other.n_quadrotors, n_balls=other.n_balls, converter=converter)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            y = output.SetFromVector(x)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()
            q = x[:3]
            qdot = x[3:]
            qddot = np.array([
                -np.sin(q[2]) / self.mass * (u[0] + u[1]) ,
                np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
                self.length / self.inertia * (u[0] - u[1])
            ])
            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((qdot, qddot)))

        def CalcMassMatrix(self, context):
            M = np.array([[self.mass, 0, 0],[0, self.mass,0],[0,0,self.inertia]])
            return M

        def CalcGravTerm(self, context):
            G = np.array([0,-self.mass*self.gravity,0])
            return G

        def CalcInputVec(self, context, u):
            x = context.get_continuous_state_vector().CopyToVector()
            q = x[:3]
            qdot = x[3:]
            InputVec = np.array([-(u[0] + u[1])*np.sin(q[2]), (u[0] + u[1])*np.cos(q[2]), self.length*(u[0] - u[1])])
            return InputVec

        def DoGetWitnessFunctions(self, context):
            return self.witness_functions

        def WitnessCollisionBall(self, context, ball_ind):
            x = context.get_continuous_state_vector().CopyToVector()
            x_ball = self.get_input_port(1 + self.n_quadrotors + ball_ind).Eval(context)
            q = x[0:3]
            q_ball = x_ball[0:2]
            return CalcClosestDistanceQuadBall(q, q_ball)

        def EventCollisionBall(self, context, event, state, ball_ind):
            x = state.get_continuous_state().CopyToVector()
            x_ball = self.get_input_port(1 + self.n_quadrotors + ball_ind).Eval(context)

            x = CalcPostCollisionStateQuadBall(x, x_ball)
            state.get_mutable_continuous_state().get_mutable_vector().SetFromVector(x)

    return Impl


Quadrotor2D = Quadrotor2D_[None]  # Default instantiation
