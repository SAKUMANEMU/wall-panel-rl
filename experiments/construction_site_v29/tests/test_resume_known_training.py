"""Evaluation recovery retains finished episodes and enforces model identity."""
import sys
from concurrent.futures import Future
import pytest
from wall_panel_rl.utils.config import ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from resume_known_training import cached_evaluate
from run_adaptation import save
import resume_known_training as resume
import json
import zipfile


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


def recovery_fixture(tmp_path, monkeypatch):
    project=tmp_path/'project';run=tmp_path/'run';output=run/'training'
    monkeypatch.setattr(resume,'ROOT',project)
    source=project/'models/current_ppo.zip';checkpoint=output/'models/ppo_32768.zip'
    for path,steps in [(source,100),(checkpoint,32868)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('data',json.dumps({'num_timesteps':steps}))
            archive.writestr('policy.optimizer.pth',b'fixture')
    for split in ('train','validation'):
        save(run/'scenes'/f'{split}_cases.json',[{'seed':split}])
    spec={'seed':42,'training':{'steps':65536,'checkpoint_interval':8192,
                              'teacher_cases_per_count':3}}
    save(output/'protocol.json',{'source_files':{},'settings':spec,
        'source_model_sha256':resume.digest(source),
        **{f'{split}_bank_sha256':resume.digest(run/'scenes'/f'{split}_cases.json')
           for split in ('train','validation')}})
    (run/'training_config.yaml').write_text(resume.yaml.safe_dump(spec))
    save(run/'runtime.json',{'frozen_site':'original height limits'})
    return output,checkpoint,spec


def test_recovery_uses_frozen_run_settings_without_project_defaults(tmp_path,monkeypatch):
    output,checkpoint,spec=recovery_fixture(tmp_path,monkeypatch)
    _,settings,runtime,_,_,completed=resume.resume_preflight(output,checkpoint)
    assert settings==spec and completed==32768
    assert runtime=={'frozen_site':'original height limits'}


def test_recovery_rejects_modified_training_scene_bank(tmp_path,monkeypatch):
    output,checkpoint,_=recovery_fixture(tmp_path,monkeypatch)
    save(output.parent/'scenes/train_cases.json',[{'seed':'changed'}])
    with pytest.raises(ValueError,match='frozen case bank changed'):
        resume.resume_preflight(output,checkpoint)
