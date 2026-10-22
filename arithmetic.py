"""Prespecified exact arithmetic and binary32 emulators; no native kernel claims."""
from fractions import Fraction as F
import struct

import numpy as np


def frac(x):
    return x if isinstance(x, F) else F(int(x)) if isinstance(x, (int, np.integer)) else F.from_float(float(x))


def pack(x):
    x = frac(x)
    return {'n': x.numerator, 'd': x.denominator, 'decimal': float(x)}


def unpack(x):
    return F(x['n'], x['d'])


def even(x):
    x = frac(x); q, r = divmod(x.numerator, x.denominator)
    return q + int(2*r > x.denominator or (2*r == x.denominator and q % 2))


def pow2(e):
    return F(2**e) if e >= 0 else F(1, 2**(-e))


def f32(x):
    """Single exact-rational round to binary32, including subnormals.

    A rational exact zero has no sign and maps to +0. The frozen scale paths
    have positive scales and integer t, so no negative exact-zero product.
    Observed native signed zeros remain preserved by float_record.
    """
    x = frac(x)
    sign = int(x < 0); a = abs(x)
    if not a:
        bits = 0
    else:
        e = a.numerator.bit_length() - a.denominator.bit_length()
        if a < pow2(e): e -= 1
        if e < -126:
            significand = even(a / pow2(-149))
            bits = significand  # 2**23 naturally carries into minimum normal.
        else:
            significand = even(a / pow2(e-23))
            if significand == 2**24:
                e += 1; significand //= 2
            if e > 127: raise OverflowError('binary32 overflow')
            bits = ((e+127) << 23) | (significand-2**23)
        bits |= sign << 31
    return np.frombuffer(struct.pack('<I', bits), dtype='<f4')[0]


def bits(x):
    return f'{struct.unpack("<I", np.asarray(x, dtype="<f4").tobytes())[0]:08x}'


def float_record(x):
    x = np.float32(x)
    assert np.isfinite(x)
    return {'value': float(x), 'bits_hex': bits(x), 'hex': float(x).hex(), 'exact': pack(x)}


def integer_reference(a, b, c, za, zb):
    """Independent Python scalar and int64 paths; no overflow or wrapping allowed."""
    a = np.asarray(a).reshape(-1); b = np.asarray(b); c = np.asarray(c)
    zb = np.asarray(zb).reshape(-1)
    assert b.shape == (len(c), len(a)) and zb.shape == c.shape
    scalar = []
    for row, bias, zero in zip(b, c, zb):
        partial = 0; lo = hi = None; absolute = 0
        for ak, bk in zip(a, row):
            product = (int(ak)-int(za))*(int(bk)-int(zero))
            absolute += abs(product); partial += product
            lo = partial if lo is None else min(lo, partial)
            hi = partial if hi is None else max(hi, partial)
            assert -(2**31) <= partial <= 2**31-1, 'INT32 partial overflow'
        total = partial + int(bias)
        assert -(2**31) <= total <= 2**31-1, 'INT32 bias-added overflow'
        assert absolute+abs(int(bias)) <= 2**31-1, 'INT32 reassociation guard'
        scalar.append({'dot': partial, 'bias': int(bias), 'total': total,
                       'partial_min': lo, 'partial_max': hi,
                       'absolute_bound_including_bias': absolute+abs(int(bias))})
    products = (a.astype(np.int64)-int(za))[None, :]*(b.astype(np.int64)-zb.astype(np.int64)[:, None])
    cumulative = products.cumsum(axis=1, dtype=np.int64)
    totals = cumulative[:, -1] + c.astype(np.int64)
    for j, row in enumerate(scalar):
        assert row['dot'] == int(cumulative[j,-1]) and row['total'] == int(totals[j])
        assert row['partial_min'] == int(cumulative[j].min()) and row['partial_max'] == int(cumulative[j].max())
        assert row['absolute_bound_including_bias'] == int(np.abs(products[j]).sum()+abs(int(c[j])))
    return scalar


def scale_paths(t, sa, sb, sy):
    """Exactly four frozen paths. Each named intermediate is binary32-rounded."""
    t, sa, sb, sy = map(frac, [t, sa, sb, sy])
    paths = {}
    def build(name, sequence):
        values = {}; records = []
        for label, operation in sequence:
            value = f32(operation(values)); values[label] = frac(value)
            records.append({'step': label, **float_record(value)})
        paths[name] = {'intermediates': records, 'output': records[-1],
                       'units': 'score' if name in ['S1','S2'] else 'code_before_zero'}
    build('S1', [('t_float', lambda v:t), ('sa_sb',lambda v:sa*sb),
                 ('score',lambda v:v['t_float']*v['sa_sb'])])
    build('S2', [('t_float',lambda v:t), ('t_sa',lambda v:v['t_float']*sa),
                 ('score',lambda v:v['t_sa']*sb)])
    build('S3', [('t_float',lambda v:t), ('sa_sb',lambda v:sa*sb),
                 ('sa_sb_div_sy',lambda v:v['sa_sb']/sy),
                 ('code_units',lambda v:v['t_float']*v['sa_sb_div_sy'])])
    build('S4', [('t_float',lambda v:t), ('sb_div_sy',lambda v:sb/sy),
                 ('sa_times_sb_div_sy',lambda v:sa*v['sb_div_sy']),
                 ('code_units',lambda v:v['t_float']*v['sa_times_sb_div_sy'])])
    return paths


def code_units_record(v, zy):
    v = frac(v); rounded = even(v); shifted = rounded+int(zy)
    code = max(-128, min(127, shifted))
    floor = v.numerator // v.denominator; threshold = F(2*floor+1, 2)
    return {'before_zero': pack(v), 'after_zero_exact': pack(v+int(zy)),
            'rounded_before_zero': rounded, 'unclipped_after_zero': shifted, 'code': code,
            'saturated': shifted != code, 'endpoint_occupancy': code in [-128,127],
            'nearest_threshold_before_zero': pack(threshold),
            'signed_threshold_distance_code_units': pack(v-threshold),
            'threshold_lower_code': floor+int(zy),
            'threshold_belongs_to_lower': floor % 2 == 0,
            'finite_cell_edges_included': (code-int(zy)) % 2 == 0}


def output_maps(x, sy, zy):
    x, sy = frac(x), frac(sy); assert sy > 0
    exact = code_units_record(x/sy, zy)
    divided = f32(x/sy)
    reciprocal = f32(1/sy); multiplied = f32(x*frac(reciprocal))
    return {'exact': exact,
            'R1': {**code_units_record(divided,zy), 'division_float32': float_record(divided)},
            'R2': {**code_units_record(multiplied,zy), 'reciprocal_float32': float_record(reciprocal),
                   'product_float32': float_record(multiplied)}}


def boundary_probes(sy, zy, lower_codes):
    rows = []
    for lower in lower_codes:
        boundary = (F(2*lower+1,2)-int(zy))*frac(sy)
        nearest = f32(boundary)
        values = [np.nextafter(nearest,np.float32(-np.inf)), nearest,
                  np.nextafter(nearest,np.float32(np.inf))]
        seen = set()
        for value in values:
            if bits(value) in seen: continue
            seen.add(bits(value)); distance = frac(value)-boundary
            rows.append({'lower_code':lower,'boundary':pack(boundary),
                         'boundary_representable':frac(nearest)==boundary,
                         'input':float_record(value),'signed_distance_score':pack(distance),
                         'side': 'below' if distance<0 else 'above' if distance>0 else 'equal'})
    return rows
