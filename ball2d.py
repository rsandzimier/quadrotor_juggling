import numpy as np

from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType, WitnessFunctionDirection, UnrestrictedUpdateEvent_
from pydrake.systems.pyplot_visualizer import PyPlotVisualizer
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.symbolic import Expression
import time
from collisions import CalcClosestDistanceBallBall, CalcPostCollisionStateBallBall, CalcClosestDistanceQuadBall, CalcPostCollisionStateBallQuad

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html


# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
@TemplateSystem.define("Ball2D_")
def Ball2D_(T):

    class Impl(LeafSystem_[T]):
        def _construct(self, n_quadrotors=0, n_balls=0, converter=None):
            # n_balls is the number of OTHER balls (besides this one)

            LeafSystem_[T].__init__(self, converter)
            self.n_quadrotors = n_quadrotors
            self.n_balls = n_balls
            # four outputs (full state)
            self.DeclareVectorOutputPort("x", BasicVector_[T](4),
                                         self.CopyStateOut)
            # two positions, two velocities
            self.DeclareContinuousState(2, 2, 0)

            # Quadrotor states as inputs
            for i in range(self.n_quadrotors):
                    self.DeclareVectorInputPort("quad_" + str(i), BasicVector_[T](6))

            # Other ball states as inputs
            for i in range(self.n_balls):
                self.DeclareVectorInputPort("ball_" + str(i), BasicVector_[T](4))

            self.witness_functions = []
            witness_callback_quad = lambda quad_ind: lambda context: self.WitnessCollisionQuad(context, quad_ind)
            event_callback_quad = lambda quad_ind: lambda context, event, state: self.EventCollisionQuad(context, event, state, quad_ind)
            for i in range(self.n_quadrotors):
                witness_function = self.MakeWitnessFunction("collision_quad_" + str(i), WitnessFunctionDirection.kPositiveThenNonPositive,
                            witness_callback_quad(i), UnrestrictedUpdateEvent_[T](event_callback_quad(i)))
                self.witness_functions.append(witness_function)
            witness_callback_ball = lambda ball_ind: lambda context: self.WitnessCollisionBall(context, ball_ind)
            event_callback_ball = lambda ball_ind: lambda context, event, state: self.EventCollisionBall(context, event, state, ball_ind)
            for i in range(self.n_balls):
                witness_function = self.MakeWitnessFunction("collision_ball_" + str(i), WitnessFunctionDirection.kPositiveThenNonPositive,
                            witness_callback_ball(i), UnrestrictedUpdateEvent_[T](event_callback_ball(i)))
                self.witness_functions.append(witness_function)

            self.radius = 0.1
            self.mass = 0.1
            self.gravity = 9.81
            self.mu = 0.3
            
            self.stiffness_ball = 10000.0
            self.damping_ball = 0.0

            self.width_quad = 0.3

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, n_quadrotors=other.n_quadrotors, n_balls=other.n_balls, converter=converter)

        def CopyStateOut(self, context, output):
            x = context.get_continuous_state_vector().CopyToVector()
            y = output.SetFromVector(x)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            q = x[:2]
            qdot = x[2:]

            qddot = np.array([0,-self.gravity])
            derivatives.get_mutable_vector().SetFromVector(
                np.concatenate((qdot, qddot)))

        def DoGetWitnessFunctions(self, context):
            return self.witness_functions

        def WitnessCollisionQuad(self, context, quad_ind):
            x = context.get_continuous_state_vector().CopyToVector()
            x_quad = self.get_input_port(quad_ind).Eval(context)
            q = x[0:2]
            q_quad = x_quad[0:3]
            return CalcClosestDistanceQuadBall(q_quad, q)

        def EventCollisionQuad(self, context, event, state, quad_ind):
            x = state.get_continuous_state().CopyToVector()
            x_quad = self.get_input_port(quad_ind).Eval(context)

            x = CalcPostCollisionStateBallQuad(x, x_quad)
            state.get_mutable_continuous_state().get_mutable_vector().SetFromVector(x)

        def WitnessCollisionBall(self, context, ball_ind):
            x = context.get_continuous_state_vector().CopyToVector()
            x_other = self.get_input_port(self.n_quadrotors + ball_ind).Eval(context)
            q = x[0:2]
            q_other = x_other[0:2]
            return CalcClosestDistanceBallBall(q, q_other)

        def EventCollisionBall(self, context, event, state, ball_ind):
            x = state.get_continuous_state().CopyToVector()
            x_other = self.get_input_port(self.n_quadrotors + ball_ind).Eval(context)

            x = CalcPostCollisionStateBallBall(x, x_other)
            state.get_mutable_continuous_state().get_mutable_vector().SetFromVector(x)

    return Impl


Ball2D = Ball2D_[None]  # Default instantiation