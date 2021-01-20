import pytest

# use relative tolerances:
RTOL = 10 ** -5

# use small number so that greek approximation is close
DELTA = 10 ** -8


@pytest.mark.parametrize('greek_name, independent_variable, option_type, invert', [('vega', 'vol', 'Call', False),
                                                                                   ('vega', 'vol', 'Put', False),
                                                                                   ('theta', 'tte', 'Call', True),
                                                                                   ('theta', 'tte', 'Put', True),
                                                                                   ('delta', 'spot', 'Put', False),
                                                                                   ('delta', 'spot', 'Call', False)])
def test_greek(greek_name, independent_variable, option_type, invert):
    """
    Test that a greek (derivative) is properly calculating the change in theoretical value with respect to it's
    independent variable.

    :param str greek_name: Greek attribute name (vega, theta, etc...)
    :param str independent_variable: name of variable to change that Greek is tracking (vol, tte, etc..)
    :param str option_type: 'Call' or 'Put'
    :param Boolean invert: True if the greek is negative (theta)
    """
    from numpy.testing import assert_allclose

    from scripts.pricer import (Call, Put)

    option = {'Call': Call, 'Put': Put}[option_type](100, 105, .2, .01, .01, .1)
    option.update_greeks()

    starting_theo = option.theo

    # Change independent variable and get change in greek
    setattr(option, independent_variable, getattr(option, independent_variable) + DELTA)
    option.update_greeks()
    new_theo = option.theo
    greek = getattr(option, greek_name)

    if invert is True:
        greek *= -1

    assert_allclose(greek, (new_theo - starting_theo) / DELTA, rtol=RTOL, atol=0)
