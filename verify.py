"""Verify the reviewed files and regenerate every numerical result offline."""
import hashlib
import json
from pathlib import Path
import tempfile

from analyze import BASE,generate


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest=json.loads((BASE/'MANIFEST.json').read_text());expected=set()
    for row in manifest['files']:
        p=(BASE/row['file']).resolve();assert p.is_relative_to(BASE)
        assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],row['file']
        expected.add(row['file'])
    ignored={'.git','.venv','__pycache__','generated','.pytest_cache'}
    actual={str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()
            and not any(part in ignored for part in p.relative_to(BASE).parts)
            and p.name not in ['MANIFEST.json','.DS_Store']}
    assert actual==expected,{'unexpected':sorted(actual-expected),'missing':sorted(expected-actual)}
    with tempfile.TemporaryDirectory(prefix='int8-arithmetic-verify-') as temp:
        generate(temp)
        regenerated={p.name:p for p in Path(temp).iterdir() if p.is_file()}
        baseline={p.name:p for p in (BASE/'results').iterdir() if p.is_file()}
        assert regenerated.keys()==baseline.keys()
        for name,p in regenerated.items():assert p.read_bytes()==baseline[name].read_bytes(),name
    result={'verified_files':len(expected),'numerical_results_regenerated_exactly':len(baseline),
            'native_model_execution':False,'downloads':False}
    return result


if __name__=='__main__':print(json.dumps(verify(),indent=2))
