from unittest import TestCase, main as run_module_suite

# This only implements enough to make the SciPy code work

class Tester(object):
    def test(self):
        pass
    def bench(self):
        pass


def assert_equal(actual, desired, err_msg='', verbose=True):
    if actual != desired:
        msg = "\nItems are not equal: %s" % (err_msg,)
        if verbose:
            msg += "\n ACTUAL: %r\n DESIRED: %r" % (actual, desired)
        raise AssertionError(msg)

def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    if abs(actual - desired) >= 0.5 * 10**(-decimal):
        msg = "\nArrays are not equal to %d decimals:\n%s" % (decimal, err_msg)
        if verbose:
            msg += "\n ACTUAL: %r\n DESIRED: %r" % (actual, desired)
        raise AssertionError(msg)

def assert_array_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    max_diff = abs(actual-desired).max()
    
    if max_diff >= 0.5 * 10**(-decimal):
        msg = "\n(fpypy) Arrays are not equal to %d decimals:\n%s" % (decimal, err_msg)
        if verbose:
            msg += "\n ACTUAL: %r\n DESIRED: %r" % (actual, desired)
        raise AssertionError(msg)
