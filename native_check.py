"""Optional CPU standalone Q/DQ replay; never executes the original classifier."""
import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import helper as H,TensorProto as T,numpy_helper as NH
import onnxruntime as ort

from analyze import BASE,load,compute
from arithmetic import f32,unpack


def exact(a,b,label):
    if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise AssertionError(label)


def check(out):
    versions={p:importlib.metadata.version(p) for p in ['numpy','onnx','onnxruntime']}
    assert versions=={'numpy':'2.5.4','onnx':'1.23.2','onnxruntime':'1.31.0'},'Use the pinned environment; other versions require a separately labeled check'
    data=load();params=data['parameters'];sy=f32(unpack(params['sy']['exact']));zy=np.array(params['zy'],np.int8)
    nodes=[H.make_node('QuantizeLinear',['scores','output_scale','output_zero'],['codes']),
           H.make_node('DequantizeLinear',['codes','output_scale','output_zero'],['logits'])]
    g=H.make_graph(nodes,'standalone_output_quantizer',[H.make_tensor_value_info('scores',T.FLOAT,[1,10])],
                   [H.make_tensor_value_info('codes',T.INT8,[1,10]),H.make_tensor_value_info('logits',T.FLOAT,[1,10])],
                   [NH.from_array(np.array(sy,np.float32),'output_scale'),NH.from_array(zy,'output_zero')])
    model=H.make_model(g,opset_imports=[H.make_opsetid('',17)],ir_version=8);onnx.checker.check_model(model)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'failed','scope':'standalone output quantizer only','versions':versions,
            'full_classifier_or_integer_dot_replay':False,'completed_calls':0}
    ort.disable_telemetry_events()
    summary,rows,flat,matches,decisions,probes=compute()
    try:
        with tempfile.TemporaryDirectory(prefix='int8-native-check-') as temp:
            temp=Path(temp);opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
            opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_profiling=True;opts.profile_file_prefix=str(temp/'profile');opts.optimized_model_filepath=str(temp/'optimized.onnx')
            s=ort.InferenceSession(model.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
            assert s.get_providers()==['CPUExecutionProvider']
            def run(vector,codes,logits,label):
                observed=s.run(['codes','logits'],{'scores':np.array([vector],np.float32)})
                exact(observed[0][0],np.array(codes,np.int8),label+'/codes')
                exact(observed[1][0],np.array(logits,np.float32),label+'/DQ')
                result['completed_calls']+=1
            for sample,row in zip(data['samples'],rows):
                run([r['native_pre']['value'] for r in sample['classes']],
                    [r['recomposed_code'] for r in sample['classes']],sample['recorded_recomposed_logits'],f'p/{sample["ordinal"]}')
                for name in ['S1','S2']:
                    ref=sample['recorded_native_scale_counterfactuals'][name]
                    run([r['scale_paths'][name]['output']['value'] for r in row['classes']],ref['codes'],ref['logits'],f'{name}/{sample["ordinal"]}')
            for i,probe in enumerate(probes):run([probe['input']['value']]*10,probe['native_codes'],probe['native_DQ'],f'boundary/{i}')
            events=json.loads(Path(s.end_profiling()).read_text())
            ops=[e['args'] for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
            assert {v['provider'] for v in ops}=={'CPUExecutionProvider'}
            counts=dict(collections.Counter(v['op_name'] for v in ops))
            assert result['completed_calls']==126 and counts=={'QuantizeLinear':126,'DequantizeLinear':126}
            optimized=onnx.load(temp/'optimized.onnx')
            assert any(n.op_type=='QuantizeLinear' for n in optimized.graph.node)
            result.update({'status':'passed','recorded_p_vectors':33,'recorded_scale_counterfactual_vectors':66,
                           'boundary_vectors':27,'CPU_operator_calls':counts})
    except BaseException as error:
        result['failure']={'type':type(error).__name__,'message':str(error)}
        out.write_text(json.dumps(result,indent=2)+'\n');raise
    out.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=BASE/'generated/native-check.json')
    print(json.dumps(check(parser.parse_args().out),indent=2))
