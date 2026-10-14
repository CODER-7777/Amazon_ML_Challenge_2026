"""Run actual CatBoost training, CLI stages, resume, packaging and reproduction."""
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from fixture_data import make_dataset
from ber.core import fold, macro_score, rows

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT.parent/'6ab10eb3b23ba_student_resource'/'student_resource'/'utils'/'validate_submission.py'
if not VALIDATOR.exists():
    VALIDATOR = ROOT/'utils'/'validate_submission.py'


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ber-e2e-')
        cls.root = Path(cls.temp.name)
        cls.data = cls.root/'dataset'
        cls.truth = make_dataset(cls.data)
        cls.work = cls.root/'artifacts'
        cls.env = dict(os.environ,PYTHONPATH=str(ROOT/'src'))
        cls.logs = []
        cls.run_cli('train','--sample-anchors','10000','--random-targets','10000','--iterations','60')
        cls.run_cli('calibrate')
        cls.run_cli('predict')
        cls.run_cli('package','--team','VALIDATION_ONLY')

    @classmethod
    def tearDownClass(cls):
        destination = os.environ.get('BER_VALIDATION_SAVE')
        if destination:
            target=Path(destination)
            shutil.copytree(cls.root,target,dirs_exist_ok=True)
            (target/'commands.log').write_text('\n\n'.join(cls.logs))
        cls.temp.cleanup()

    @classmethod
    def run_cli(cls,stage,*extra,work=None,success=True,source=None):
        command=[sys.executable,'-m','ber','--data',str(cls.data),'--work',str(work or cls.work),'--threads','2','--batch-size','23',*map(str,extra),stage]
        env=cls.env if source is None else dict(cls.env,PYTHONPATH=str(source))
        result=subprocess.run(command,env=env,text=True,capture_output=True,timeout=180)
        cls.logs.append(' '.join(command)+'\n'+result.stdout+result.stderr)
        if (result.returncode==0)!=success:
            raise AssertionError(f'Unexpected return code {result.returncode}:\n{result.stdout}\n{result.stderr}')
        return result

    def fresh_work(self,name,calibrated=True):
        target=self.root/name
        target.mkdir()
        for filename in ['model.cbm','model_config.json','training_report.json']+(['calibration.json'] if calibrated else []):
            shutil.copy2(self.work/filename,target/filename)
        return target

    def test_official_validator_and_country_coverage(self):
        result=subprocess.run([sys.executable,str(VALIDATOR),'--matching',str(self.work/'output/matching_results.tsv'),
                               '--candidate',str(self.work/'output/candidate_pairs.tsv'),'--test-dir',str(self.data/'test'),'--check-ids'],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('PASS',result.stdout)
        predictions={r['source1_entity_id']:set(filter(None,r['matched_entity_ids'].split(','))) for r in rows(self.work/'output/matching_results.tsv')}
        self.assertEqual(set(predictions),set(self.truth['test']))
        france={r['entity_id'] for r in rows(self.data/'test/test_source1.tsv') if r['country']=='France'}
        self.assertTrue(france)
        self.assertTrue(france<=predictions.keys())
        self.assertTrue(any(predictions[k] for k in france),'French anchors should be evaluated, not silently all discarded.')
        self.assertTrue(any(len(v)>1 for v in predictions.values()),'One-to-many matches must survive export.')

    def test_calibration_metric_matches_independent_set_formula(self):
        report=json.loads((self.work/'calibration.json').read_text())
        expected={k:v for k,v in self.truth['train'].items() if fold(k)=='valid'}
        predicted={}
        with sqlite3.connect(self.work/'scores_train.sqlite') as db:
            for anchor,target in db.execute('SELECT anchor,target FROM pairs WHERE selected=1 AND probability>=? AND margin>=?',(report['threshold'],report['margin'])):
                predicted.setdefault(anchor,set()).add(target)
        self.assertAlmostEqual(report['macro_f05'],macro_score(expected,predicted),places=12)
        self.assertEqual(report['validation_anchors'],len(expected))
        self.assertTrue(report['validation_singletons']>0)

    def test_calibration_resume_does_not_publish_partial_results(self):
        work=self.fresh_work('calibration_resume',calibrated=False)
        self.run_cli('calibrate','--max-targets','31',work=work)
        self.assertFalse((work/'calibration.json').exists())
        self.run_cli('calibrate',work=work)
        self.assertEqual(json.loads((work/'calibration.json').read_text()),json.loads((self.work/'calibration.json').read_text()))

    def test_prediction_resume_and_inference_audit(self):
        work=self.fresh_work('prediction_resume')
        self.run_cli('predict','--max-targets','17',work=work)
        self.assertFalse((work/'output/matching_results.tsv').exists())
        self.run_cli('package','--team','VALIDATION_ONLY',work=work,success=False)
        self.run_cli('predict',work=work)
        for filename in ['matching_results.tsv','candidate_pairs.tsv']:
            self.assertEqual((work/'output'/filename).read_bytes(),(self.work/'output'/filename).read_bytes())
        candidates={r['source1_entity_id']:set(filter(None,r['candidate_entity_ids'].split(','))) for r in rows(work/'output/candidate_pairs.tsv')}
        with sqlite3.connect(work/'scores_test.sqlite') as db:
            actual={}
            for anchor,target in db.execute('SELECT anchor,target FROM pairs'):
                actual.setdefault(anchor,set()).add(target)
        self.assertEqual({k:v for k,v in candidates.items() if v},actual)

    def test_zip_reproduces_identical_outputs(self):
        unpacked=self.root/'unpacked'
        with zipfile.ZipFile(self.work/'VALIDATION_ONLY_submission.zip') as archive:
            names=set(archive.namelist())
            for name in ['Documentation_template.md','output/matching_results.tsv','output/candidate_pairs.tsv','code/business_entity_resolution/README.md','code/business_entity_resolution/requirements.txt']:
                self.assertIn(name,names)
            archive.extractall(unpacked)
        code=unpacked/'code/business_entity_resolution'
        self.run_cli('predict',work=code/'artifacts',source=code/'src')
        for filename in ['matching_results.tsv','candidate_pairs.tsv']:
            self.assertEqual((code/'artifacts/output'/filename).read_bytes(),(unpacked/'output'/filename).read_bytes())

    def test_packaging_rejects_modified_outputs(self):
        work=self.root/'tampered_output'
        shutil.copytree(self.work,work)
        path=work/'output/matching_results.tsv'
        with path.open('a') as f:
            f.write('S1-not-real\tS2-not-real\n')
        self.run_cli('package','--team','INVALID',work=work,success=False)

    def test_packaging_rejects_changed_threshold(self):
        work=self.root/'changed_threshold'
        shutil.copytree(self.work,work)
        path=work/'calibration.json'
        changed=json.loads(path.read_text())
        changed['threshold']=.99999
        path.write_text(json.dumps(changed))
        self.run_cli('package','--team','INVALID',work=work,success=False)

    def test_stale_model_is_rejected(self):
        work=self.fresh_work('changed_model')
        with (work/'model.cbm').open('ab') as f:
            f.write(b'changed')
        result=self.run_cli('predict',work=work,success=False)
        self.assertIn('Model changed',result.stderr)


if __name__=='__main__':
    unittest.main()
