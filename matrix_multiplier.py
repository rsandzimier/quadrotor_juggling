from pydrake.systems.framework import BasicVector_, LeafSystem_
from pydrake.systems.scalar_conversion import TemplateSystem

import numpy as np

# Note: In order to use the Python system with drake's autodiff features, we
# have to add a little "TemplateSystem" boilerplate (for now).  For details,
# see https://drake.mit.edu/pydrake/pydrake.systems.scalar_conversion.html


# TODO(russt): Clean this up pending any resolutions on
#  https://github.com/RobotLocomotion/drake/issues/10745
@TemplateSystem.define("MatrixMultiplier_")
def MatrixMultiplier_(T):
    class Impl(LeafSystem_[T]):
        def _construct(self, left_mat_shape, right_mat_shape, converter=None):
            LeafSystem_[T].__init__(self, converter)
            # left_mat_shape and right_mat_shape are tuples of length 2 representing the shape of the matrices to be multiplied
            assert isinstance(left_mat_shape, (list,tuple)) and len(left_mat_shape) == 2, "left_mat_shape must be list or tuple of length 2"
            assert isinstance(right_mat_shape, (list,tuple)) and len(right_mat_shape) == 2, "right_mat_shape must be list or tuple of length 2"
            self.left_mat_shape = left_mat_shape
            self.right_mat_shape = right_mat_shape
            self.num_rows_left, self.num_cols_left = self.left_mat_shape
            self.num_rows_right, self.num_cols_right = self.right_mat_shape
            self.size_left = self.num_rows_left*self.num_cols_left
            self.size_right = self.num_rows_right*self.num_cols_right
            self.size_output = self.num_rows_left*self.num_cols_right
            assert self.num_cols_left == self.num_rows_right, "Dimension mismatch. left_mat_shape[1] and right_mat_shape[0] must be equal"

            self.DeclareVectorInputPort("left_mat", BasicVector_[T](self.size_left))
            self.DeclareVectorInputPort("right_mat", BasicVector_[T](self.size_right))

            self.DeclareVectorOutputPort("output_mat", BasicVector_[T](self.size_output), self.CalcMatrixProduct)

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, other.left_mat_shape, other.right_mat_shape, converter=converter)

        def CalcMatrixProduct(self, context, output):
            left_mat = self.get_input_port(0).Eval(context).reshape(self.num_rows_left, self.num_cols_left)
            right_mat = self.get_input_port(1).Eval(context).reshape(self.num_rows_right, self.num_cols_right)
            output.SetFromVector(np.dot(left_mat, right_mat).reshape(-1))

    return Impl


MatrixMultiplier = MatrixMultiplier_[None]  # Default instantiation