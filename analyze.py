"""Reproduce arithmetic from recorded observations; no model inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from arithmetic import (frac,pack,unpack,float_record,scale_paths,output_maps,
                        code_units_record,boundary_probes)

BASE=Path(__file__).resolve().parent
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def read(path):return json.loads(path.read_text())


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load():
    data=read(BASE/'data/observations.json')
    assert len(data['samples'])==33 and len(data['parameters']['sb'])==10
    for i,sample in enumerate(data['samples']):
        assert sample['ordinal']==i and len(sample['classes'])==10
        assert [r['class'] for r in sample['classes']]==list(range(10))
        for row in sample['classes']:
            integer=row['integer']
            assert integer['dot']+integer['bias']==integer['total']
            assert -(2**31)<=integer['partial_min']<=integer['partial_max']<=2**31-1
            assert abs(integer['total'])<=2**31-1 and integer['absolute_bound_including_bias']<=2**31-1
            assert float_record(np.float32(row['native_pre']['value']))==row['native_pre']
    return data


def compute():
    data=load();params=data['parameters'];sa=unpack(params['sa']['exact']);sy=unpack(params['sy']['exact']);zy=params['zy']
    sb=[unpack(v['exact']) for v in params['sb']]
    rows=[];flat=[];matches=[];decisions=[];summaries={};score_matches={'S1':0,'S2':0}
    for sample in data['samples']:
        trace=[];vectors={}
        for recorded in sample['classes']:
            j=recorded['class'];integer=recorded['integer'];r=frac(integer['total'])*sa*sb[j]
            p=unpack(recorded['native_pre']['exact']);paths=scale_paths(integer['total'],sa,sb[j],sy)
            pmaps=output_maps(p,sy,zy);ideal=output_maps(r,sy,zy)['exact']
            row={**recorded,'r':pack(r),'v':pack(r/sy),'native_pre_minus_r':pack(p-r),
                 'native_pre_maps':pmaps,'ideal_r_map':ideal,'scale_paths':paths}
            codes={f'p_{name}':value['code'] for name,value in pmaps.items()};codes['r_exact']=ideal['code']
            for name,path in paths.items():
                value=unpack(path['output']['exact'])
                if name in ['S1','S2']:
                    path['maps']=output_maps(value,sy,zy)
                    path['matches_native_pre_bits']=path['output']['bits_hex']==recorded['native_pre']['bits_hex']
                    score_matches[name]+=int(path['matches_native_pre_bits'])
                    for method,mapping in path['maps'].items():codes[f'{name}_{method}']=mapping['code']
                    path['native_code']=sample['recorded_native_scale_counterfactuals'][name]['codes'][j]
                    codes[f'{name}_native']=path['native_code']
                else:
                    path['rounding']=code_units_record(value,zy);codes[name]=path['rounding']['code']
            assert len(codes)==14
            for name,code in codes.items():
                summary=summaries.setdefault(name,{'class_values':330,'original_code_matches':0,'recomposed_code_matches':0,
                                                  'full_original_vectors':0,'full_recomposed_vectors':0})
                summary['original_code_matches']+=int(code==row['original_code'])
                summary['recomposed_code_matches']+=int(code==row['recomposed_code'])
                vectors.setdefault(name,[]).append(code)
                matches.append({'ordinal':sample['ordinal'],'split':sample['split'],'index':sample['index'],
                                'class_id':j,'path':name,'code':code,'original_code':row['original_code'],
                                'recomposed_code':row['recomposed_code'],'matches_original':code==row['original_code'],
                                'matches_recomposed':code==row['recomposed_code']})
            flat.append({'ordinal':sample['ordinal'],'split':sample['split'],'index':sample['index'],
                         'class_id':j,'class_name':CLASSES[j],'true_class':sample['label'],**integer,
                         'r':float(r),'p':float(p),'p_minus_r':float(p-r),
                         'original_code':row['original_code'],'recomposed_code':row['recomposed_code'],
                         'p_threshold_distance_score':float(unpack(pmaps['exact']['signed_threshold_distance_code_units'])*sy),
                         'r_threshold_distance_score':float(unpack(ideal['signed_threshold_distance_code_units'])*sy),**codes})
            trace.append(row)
        original=[r['original_code'] for r in trace];recomposed=[r['recomposed_code'] for r in trace]
        for name,vector in vectors.items():
            summaries[name]['full_original_vectors']+=int(vector==original)
            summaries[name]['full_recomposed_vectors']+=int(vector==recomposed)
        all_vectors={**vectors,'original':original,'recomposed':recomposed,
                     'native_pre':[unpack(r['native_pre']['exact']) for r in trace],
                     'exact_r':[unpack(r['r']) for r in trace]}
        for name,vector in all_vectors.items():
            top=max(vector);ties=[j for j,v in enumerate(vector) if v==top]
            decisions.append({'ordinal':sample['ordinal'],'split':sample['split'],'index':sample['index'],'path':name,
                              'selected_class':ties[0],'tied_classes':','.join(map(str,ties)),
                              'correct':ties[0]==sample['label'],'changes_original_class':ties[0]!=int(np.argmax(original))})
        rows.append({'ordinal':sample['ordinal'],'classes':trace})
    observed_boundaries=read(BASE/'data/boundary-observations.json')['records']
    probes=boundary_probes(sy,zy,data['boundary_lower_codes'])
    assert len(probes)==len(observed_boundaries)==27
    boundary_summary={name:{'vectors':27,'native_code_matches':0} for name in ['exact','R1','R2']}
    for probe,saved in zip(probes,observed_boundaries):
        assert probe['lower_code']==saved['lower_code'] and probe['input']==saved['input']
        probe['native_codes']=saved['native_codes'];probe['native_DQ']=saved['native_DQ']
        probe['reference_maps']=output_maps(unpack(probe['input']['exact']),sy,zy)
        for name,mapping in probe['reference_maps'].items():
            boundary_summary[name]['native_code_matches']+=int(all(v==mapping['code'] for v in saved['native_codes']))
    result={'scope':{'inputs':33,'class_values':330,'target_index':4054,'failure_selected':True},
            'path_summary':summaries,'scale_score_bit_matches_out_of_330':score_matches,
            'boundary_summary':boundary_summary,'native_values_are_recorded_observations':True,
            'integer_dot_products_not_recomputed_from_features':True,'full_model_not_executed':True}
    return result,rows,flat,matches,decisions,probes


def generate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary,rows,flat,matches,decisions,probes=compute()
    write(out/'analysis.json',summary);write(out/'target.json',rows[-1])
    write(out/'boundary-comparison.json',{'records':probes})
    write_csv(out/'all-classes.csv',flat);write_csv(out/'code-match-matrix.csv',matches)
    write_csv(out/'path-decisions.csv',decisions)
    horse=rows[-1]['classes'][7]
    table=['| Calculation | Before integer rounding and zero point | Rounded integer | Output code |',
           '| --- | ---: | ---: | ---: |']
    for label,mapping in [('Exact scaled reference',horse['ideal_r_map']),
                          ('Observed float-output score, exact division',horse['native_pre_maps']['exact']),
                          ('Folded-scale candidates S3/S4',horse['scale_paths']['S3']['rounding'])]:
        table.append(f'| {label} | {mapping["before_zero"]["decimal"]:.13f} | {mapping["rounded_before_zero"]} | {mapping["code"]} |')
    table+=['','Zero point: -80. Values in this table are reference/candidate arithmetic, not hidden native intermediates.','']
    (out/'article-table.md').write_text('\n'.join(table))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=BASE/'generated/analysis')
    out=parser.parse_args().out.resolve()
    assert out not in [BASE,BASE/'data',BASE/'results'] and not out.is_relative_to(BASE/'data') and not out.is_relative_to(BASE/'results')
    print(json.dumps(generate(out),indent=2))
