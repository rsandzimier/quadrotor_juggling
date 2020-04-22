import numpy as np

from pydrake.systems.framework import BasicVector_, LeafSystem_, PortDataType
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.symbolic import Expression
from collisions import CalcSignedInterferenceQuadBall, CalcClosestLocationQuadBall


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
            f = np.zeros(3, dtype=q.dtype)
            if not isinstance(q[0], Expression): 

                # calculate collisions with quadrotors

                # calculate forces and moments from collision with balls
                for i in range(1+self.n_quadrotors, 1 + self.n_quadrotors + self.n_balls):
                    x_ball_i = self.EvalVectorInput(context,i).CopyToVector()
                    q_ball_i = x_ball_i[:2]
                    qdot_ball_i = x_ball_i[2:]

                    signed_interference = CalcSignedInterferenceQuadBall(q, q_ball_i)
                    force_location = CalcClosestLocationQuadBall(q, q_ball_i)
                    # add elastic force
                    force_vec = self.stiffness_quad*signed_interference
                    moment = np.cross(force_location, force_vec).reshape(-1)
                    f += np.array([force_vec[0], force_vec[1], moment[0]])

            qddot = np.array([
                -np.sin(q[2]) / self.mass * (u[0] + u[1]) ,
                np.cos(x[2]) / self.mass * (u[0] + u[1]) - self.gravity,
                self.length / self.inertia * (u[0] - u[1])
            ]) + f/np.array([self.mass, self.mass, self.inertia])
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

    return Impl


Quadrotor2D = Quadrotor2D_[None]  # Default instantiation
