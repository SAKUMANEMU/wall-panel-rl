"""Evaluation recovery retains finished episodes and enforces model identity."""
import sys
from concurrent.futures import Future
import pytest
from wall_panel_rl.utils.config import ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from resume_known_training import cached_evaluate
from run_adaptation import save


def row(seed):
    return {'seed':seed,'obstacle_count':2,'reason':'success','passage_count':2,
            'position_m':.001,'orientation_deg':.1}


class FakePool:
    def __init__(self):self.submitted=[]
    def submit(self,function,job):
        seed=job[1]['scene']['seed'];self.submitted.append(seed)
        f=Future();f.set_result(row(seed));return f


class FakeStatus:
    def update(self,*args,**kwargs):pass


def test_resume_submits_only_missing_validation_cases(tmp_path):
    checkpoint=tmp_path/'model.zip';checkpoint.write_bytes(b'fixed checkpoint')
    saved=row(11)
    save(tmp_path/'evaluations/ppo_8192.json',{'model':str(checkpoint),'mode':'enhanced',
         'complete':False,'rows':[saved]})
    cases=[{'scene':{'seed':seed}} for seed in [11,12,13]]
    pool=FakePool()
    result=cached_evaluate(pool,checkpoint,cases,tmp_path,FakeStatus(),'ppo_8192')
    assert pool.submitted==[12,13]
    assert result['complete'] and result['metrics']['all']['episodes']==3
    assert result['rows'][0]==saved


def test_resume_rejects_a_different_model_cache(tmp_path):
    checkpoint=tmp_path/'model.zip';checkpoint.write_bytes(b'fixed checkpoint')
    save(tmp_path/'evaluations/ppo_8192.json',{'model':str(tmp_path/'other.zip'),
         'mode':'enhanced','complete':False,'rows':[row(11)]})
    with pytest.raises(ValueError,match='different model'):
        cached_evaluate(FakePool(),checkpoint,[{'scene':{'seed':11}}],tmp_path,FakeStatus(),'ppo_8192')
