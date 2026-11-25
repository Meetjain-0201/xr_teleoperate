"""--home-q parsing (DIM-752): default is upstream zeros; anything malformed fails fast."""

import numpy as np
import pytest

from teleop.utils.dimenso_home_pose import parse_home_q


def test_default_is_upstream_zero_home():
    np.testing.assert_array_equal(parse_home_q(None), np.zeros(14))
    np.testing.assert_array_equal(parse_home_q(""), np.zeros(14))


def test_fourteen_values_parse_in_order():
    q = parse_home_q(",".join(str(i / 10) for i in range(14)))
    np.testing.assert_allclose(q, np.arange(14) / 10)


@pytest.mark.parametrize("bad", ["1,2,3", ",".join(["0"] * 15), ",".join(["0"] * 13 + ["nan"]), "a," * 13 + "b"])
def test_malformed_home_is_refused(bad):
    with pytest.raises(ValueError):
        parse_home_q(bad)
