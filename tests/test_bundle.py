from fractions import Fraction as F
import unittest

import numpy as np

from arithmetic import (frac,even,f32,bits,integer_reference,
    scale_paths,output_maps,boundary_probes,unpack)
def require_exact(a,b,label):
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),label


class Arithmetic(unittest.TestCase):
    def test_signed_centering_transpose_and_bias(self):
        rows=integer_reference(np.array([-2,3],np.int8),np.array([[1,-1],[-2,4]],np.int8),
                               np.array([7,-3],np.int32),-1,np.array([0,1],np.int8))
        self.assertEqual([r['dot'] for r in rows],[-5,15])
        self.assertEqual([r['total'] for r in rows],[2,12])

    def test_partial_overflow(self):
        with self.assertRaisesRegex(AssertionError,'partial'):
            integer_reference(np.full(140000,127),np.full((1,140000),127),[0],0,[0])

    def test_bias_overflow(self):
        with self.assertRaisesRegex(AssertionError,'bias-added'):
            integer_reference([1],[[1]],[2**31-1],0,[0])

    def test_reassociation_bound(self):
        a=np.full(140000,127); b=np.full((1,140000),127);b[:,1::2]=-127
        with self.assertRaisesRegex(AssertionError,'reassociation'):
            integer_reference(a,b,[0],0,[0])

    def test_nearest_even_negative_ties(self):
        self.assertEqual([even(F(n,2)) for n in [-5,-3,-1,1,3,5]],[-2,-2,0,0,2,2])

    def test_zero_added_after_rounding(self):
        for zero in [-3,-2,0,1,2]:
            self.assertEqual(output_maps(F(1,2),1,zero)['exact']['code'],zero)
        self.assertNotEqual(even(F(1,2))+1,even(F(1,2)+1))

    def test_clipping_vs_endpoint(self):
        self.assertFalse(output_maps(127,1,0)['exact']['saturated'])
        self.assertTrue(output_maps(128,1,0)['exact']['saturated'])
        self.assertTrue(output_maps(-129,1,0)['exact']['saturated'])

    def test_binary32_midpoint_ties(self):
        self.assertEqual(bits(f32(F(1)+F(1,2**24))),'3f800000')
        self.assertEqual(bits(f32(F(1)+F(3,2**24))),'3f800002')
        self.assertEqual(bits(f32(-(F(1)+F(1,2**24)))),'bf800000')

    def test_subnormal_and_normal_boundary(self):
        self.assertEqual(bits(f32(F(1,2**150))),'00000000')
        self.assertEqual(bits(f32(F(3,2**150))),'00000002')
        self.assertEqual(bits(f32(-F(1,2**151))),'80000000')
        self.assertEqual(bits(f32(F(1,2**126)-F(1,2**150))),'00800000')

    def test_synthetic_numpy_float32_crosscheck(self):
        values=[np.float32(x) for x in [-1001.3,-.1,0,.03,.7,1,3.25,16777215]]
        for a in values:
            for b in values:
                actual=f32(frac(a)*frac(b));expected=np.float32(a*b)
                if actual!=0 or expected!=0:self.assertEqual(bits(actual),bits(expected))
                if b:
                    # Exact zero has no sign. Ignore signed zero only here.
                    actual=f32(frac(a)/frac(b));expected=np.float32(a/b)
                    if actual!=0 or expected!=0:self.assertEqual(bits(actual),bits(expected))

    def test_four_scale_orders_match_numpy(self):
        for t in [-123456789,0,123456789]:
            a,b,s=map(np.float32,[.02,.03,.05]);v=np.float32(t)
            expected=[np.float32(v*np.float32(a*b)),np.float32(np.float32(v*a)*b),
                      np.float32(v*np.float32(np.float32(a*b)/s)),
                      np.float32(v*np.float32(a*np.float32(b/s)))]
            paths=scale_paths(t,a,b,s)
            self.assertEqual(list(paths),['S1','S2','S3','S4'])
            self.assertEqual([p['output']['bits_hex'] for p in paths.values()],[bits(x) for x in expected])

    def test_boundary_bracketing_and_exactness(self):
        for scale in [np.float32(.125),np.float32(.13)]:
            rows=boundary_probes(scale,1,[-129,-1,0,16,127])
            self.assertEqual(len(rows),15)
            for start in range(0,15,3):
                distances=[unpack(r['signed_distance_score']) for r in rows[start:start+3]]
                self.assertTrue(distances[0]<0<distances[-1])
                for row in rows[start:start+3]:
                    self.assertEqual(row['side']=='equal',unpack(row['signed_distance_score'])==0)

    def test_reference_maps_float32_intermediates(self):
        for x in [-2.25,-.65,0,.3,2.25]:
            x,s=map(np.float32,[x,.13]);maps=output_maps(x,s,-3)
            division=np.float32(x/s);recip=np.float32(np.float32(1)/s);product=np.float32(x*recip)
            self.assertEqual(maps['R1']['division_float32']['bits_hex'],bits(division))
            self.assertEqual(maps['R2']['product_float32']['bits_hex'],bits(product))

    def test_replay_rejects_changed_value_or_type(self):
        with self.assertRaises(AssertionError):require_exact(np.array([16],np.int8),np.array([17],np.int8),'replay')
        with self.assertRaises(AssertionError):require_exact(np.array([16],np.int8),np.array([16],np.int32),'type')


class Artifact(unittest.TestCase):
    def test_all_recorded_results_regenerate(self):
        from verify import verify
        result=verify()
        self.assertEqual(result['numerical_results_regenerated_exactly'],7)
        self.assertFalse(result['native_model_execution'])


if __name__=='__main__': unittest.main()
