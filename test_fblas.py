import unittest

import fblas
m = fblas
#from scipy.linalg.blas import fblas as m

import numpy

# Use only tests where "near enough" is with 0.00000000001.

def assert_near(left, right, msg):
    assert type(left) == type(right), (left, right, type(left), type(right))
    if isinstance(left, int):
        assert left == right
    elif isinstance(left, float):
        assert abs(left - right) < 0.00000000001, (left, right)
    elif isinstance(left, tuple):
        assert len(left) == len(right), (msg, left, right)
        for i, (l, r) in enumerate(zip(left, right)):
            assert_near(l, r, msg + (" item %d" % i))
    elif isinstance(left, complex):
        assert_near(left.real, right.real, msg + " (real)")
        assert_near(left.imag, right.imag, msg + " (imag)")
    elif isinstance(left, numpy.ndarray):
        assert left.shape == right.shape, (msg, left, right)
        if len(left.shape) == 0:
            assert left == right, (msg, left, right)
        else:
            for i, (l, r) in enumerate(zip(left, right)):
                assert_near(l, r, msg + (" array item %d" % i))
    elif isinstance(left, (numpy.float32, numpy.float64)):
        assert abs(left - right).max() < 0.00000000001, (left, right)
    else:
        raise AssertionError( (msg, left, right, type(left), type(right)) )
def E(left, right):
    assert_near(left, right, "")


def compare(test_func):
    for i, (pypy, py) in enumerate(zip(test_func(fblas), test_func(fblas))):
        assert_near(pypy, py, "Test %d" % (i+1,))

class CBlasTestCase(unittest.TestCase):
    def test_srotg(self):
        E(m.srotg(3,4), (0.60000002384185791, 0.80000001192092896))
        E(m.srotg(3.0,9.0), (0.31622776389122009, 0.94868332147598267))
        E(m.srotg(-7.0, 11.0), (-0.53687548637390137, 0.84366148710250854))

    def test_drotg(self):
        E(m.drotg(3,4), (0.60000000000000009, 0.80000000000000016))
        E(m.drotg(3.0,9.0), (0.31622776601683794, 0.94868329805051377))
        E(m.drotg(-7.0, 11.0), (-0.53687549219315922, 0.84366148773210736))

    def test_crotg(self):
        E(m.crotg(3,4), ((0.60000002384185791+0j), (0.80000001192092896+0j)))
        E(m.crotg(3.0,9.0), ((0.31622776389122009+0j), (0.94868332147598267+0j)))
        E(m.crotg(-7.0, 11.0), ((0.53687548637390137+0j), (-0.84366148710250854+0j)))
        E(m.crotg(3+2j, 4-1j), ((0.65828061103820801+0j), (0.50636976957321167+0.55700665712356567j)))

    def test_zrotg(self):
        E(m.zrotg(3,4), ((0.60000000000000009+0j), (0.80000000000000016+0j)))
        E(m.zrotg(3.0,9.0), ((0.31622776601683794+0j), (0.94868329805051377+0j)))
        E(m.zrotg(-7.0,11.0), ((0.53687549219315922+0j), (-0.84366148773210736+0j)))
        E(m.zrotg(3+2j, 4-1j), ((0.65828058860438332+0j), (0.50636968354183332+0.5570066518960165j)))

    def test_srotmg(self):
        E(m.srotmg(1,2,3,4), numpy.array([1., 0.375, 0.0, 0., 0.75], "f"))
        E(m.srotmg(1,9.4,-3.2,7.2), numpy.array([1., -0.04728133,  0., 0., -0.44444448], "f"))

    def test_drotmg(self):
        E(m.drotmg(1,2,3,4), numpy.array([1., 0.375, 0.0, 0., 0.75], "d"))
        E(m.drotmg(1,9.4,-3.2,7.2), numpy.array([1., -0.047281323877068557,  0., 0., -0.44444444444444448], "d"))

    def test_srot(self):
        E(m.srot(1,2,3,4), (numpy.array(11.0, dtype="f"), numpy.array(2.0, dtype="f")))
        E(m.srot(1,9.4,-3.2,7.2), (numpy.array(64.479995727539062, dtype="f"),
                                   numpy.array(-37.279998779296875, dtype="f")))
        E(m.srot([1,1],[2, 9.4],3,4), (numpy.array([ 11., 40.59999847], dtype="f"),
                                       numpy.array([  2., 24.19999886], dtype="f")))

    def test_srot_n(self):
        E(m.srot([1,1],[2, 9.4],3,4, n=1), (numpy.array([ 11., 1.0], dtype="f"),
                                            numpy.array([  2., 9.4], dtype="f")))

    def test_srot_offs(self):
        E(m.srot([1,1,4,9],[2, 9.4,3,1],0.5,0.2),
          (numpy.array([ 0.89999998,  2.38000011,  2.5999999 ,  4.69999981], "f"),
           numpy.array([ 0.80000001,  4.5       ,  0.69999999, -1.30000007], "f")))
        E(m.srot([1,1,4,9],[2, 9.4,3,1],0.5,0.2, offx=0, offy=0),
          (numpy.array([ 0.89999998,  2.38000011,  2.5999999 ,  4.69999981], "f"),
           numpy.array([ 0.80000001,  4.5       ,  0.69999999, -1.30000007], "f")))
        E(m.srot([1,1,4,9],[2, 9.4,3,1],0.5,0.2, offx=1, offy=1),
          (numpy.array([ 1,  2.38000011,  2.5999999 ,  4.69999981], "f"),
           numpy.array([ 2,  4.5       ,  0.69999999, -1.30000007], "f")))
        E(m.srot([1,1,4,9],[2, 9.4,3,1],0.5,0.2, offx=2),
          (numpy.array([ 1.,  1., 2.4000001, 6.38000011], dtype="f"),
           numpy.array([ 0.19999999, 2.89999962, 3. , 1.], dtype="f")))

    def test_srot_inc(self):
        E(m.srot([1,1,4,9],[2, 9.4,3,1],0.5,0.2, incx=1, incy=1),
          (numpy.array([ 0.89999998,  2.38000011,  2.5999999 ,  4.69999981], "f"),
           numpy.array([ 0.80000001,  4.5       ,  0.69999999, -1.30000007], "f")))

        E(m.srot([1,1,4,9],[2, 9.4,3,1],0.5,0.2, incx=2, incy=2),
          (numpy.array([ 0.89999998,  1.0, 2.5999999 , 9.0], "f"),
           numpy.array([ 0.80000001,  9.4, 0.69999999, 1.0], "f")))

        E(m.srot([1,1,4,9],[2, 9.4,3,1],0.5,0.2, incx=3, incy=2),
          (numpy.array([ 0.89999998,  1.0, 4.0 , 9.0], "f"),
           numpy.array([ 0.80000001,  9.4, 3.0, 1.0], "f")))

    def test_srot_overwrite(self):
        x = numpy.array([1,1,4,9], "f")
        y = numpy.array([2, 9.4, 3, 1], "f")
        x2, y2 = m.srot(x, y, 0.5, 0.2, overwrite_x=True)
        assert x is x2
        assert y is not y2
        
        x = numpy.array([1,1,4,9], "f")
        y = numpy.array([2, 9.4, 3, 1], "f")
        x2, y2 = m.srot(x, y, 0.5, 0.2, overwrite_y=True)
        assert x is not x2
        assert y is y2


    def _raise_m_error(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception, x:
            return x
        raise AssertionError("Did not fail", func, args, kwargs)
    
    def test_srot_errors(self):
        x = numpy.array([1,1,4,9], "f")
        y = numpy.array([2, 9.4, 3, 1], "f")
        e = self._raise_m_error(m.srot, x, y, incx=0)
        assert isinstance(e, TypeError), (e, type(e))
        # f2py
        #assert "Required argument 'c' (pos 3) not found" in str(e), str(e)
        # f2pypy
        #assert "srot() takes at least 4 non-keyword arguments (2 given)" in str(e), str(e)
        x = numpy.array([1,1,4,9], "f")
        y = numpy.array([2, 9.4, 3, 1], "f")
        e = self._raise_m_error(m.srot, x, y, 2.0, 3.3, incx=0)
        # f2py
        #assert "(incx>0||incx<0) failed for 3rd keyword incx: srot:incx=0" in str(e), str(e)
        # f2pypy
        #assert "(incx>0||incx<0) failed for argument incx: incx=0" in str(e), str(e)
        assert "(incx>0||incx<0) failed for " in str(e), str(e)
        assert "incx=0" in str(e), str(e)
                                       
    def test_drot(self):
        E(m.drot(1,2,3,4), (numpy.array(11.0, dtype="d"), numpy.array(2.0, dtype="d")))
        E(m.drot(1,9.4,-3.2,7.2), (numpy.array(64.480000000000004, dtype="d"),
                                   numpy.array(-37.280000000000001, dtype="d")))
        E(m.drot([1,1],[2, 9.4],3,4), (numpy.array([ 11., 40.600000000000001], dtype="d"),
                                       numpy.array([  2., 24.200000000000003], dtype="d")))


    def test_snrm2(self):
        x = numpy.array([1,1,4,9], "f")
        E(m.snrm2(x), float((numpy.array([1+1+16+81], "f")**0.5)))

    def test_dnrm2(self):
        x = numpy.array([1,1,4,9], "d")
        E(m.dnrm2(x), float((numpy.array([1+1+16+81], "d")**0.5)))

if __name__ == "__main__":
    unittest.main()
    
