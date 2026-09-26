"""Deprecated location of :mod:`hndl.relations`, kept so existing imports work.

Import shape relation helpers from ``hndl.relations``, the public module.
"""

from ..relations import (broadcast, conv_axis, conv_input_range, conv_output,  # noqa: F401
                         conv_transpose_axis, conv_transpose_input, conv_transpose_output,
                         elementwise_join, spatial)
