import numpy as np
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config, ROOT
from wall_panel_rl.utils.metrics import capture_scenario


def test_velocity_observation_and_replay():
    c=load_config(ROOT/'configs/ppo_dynamic_clean.yaml')
    a,b=WallPanelEnv(c),WallPanelEnv(c)
    try:
        obs,_=a.reset(seed=10002)
        scene=capture_scenario(a,10002)
        replay,_=b.reset(seed=10002,options={'scenario':scene})
        assert obs.shape==(70,) and np.array_equal(obs,replay)
        slots=obs[20:].reshape(5,10)
        before=np.array([p.copy() for p,d in a.obstacles])
        a.move_obstacles(.00001)
        after=np.array([p.copy() for p,d in a.obstacles])
        assert np.allclose(slots[:,6:9],(after-before)/.00001,atol=1e-5)
        assert np.all(slots[:,9]==1)
        a.move_obstacles(0)
        for _ in range(3):
            oa,ra,da,ta,ia=a.step(np.zeros(7))
            ob,rb,db,tb,ib=b.step(np.zeros(7))
            assert np.allclose(oa,ob) and np.isclose(ra,rb)
        b.config['obstacles']['freeze_motion']=True
        assert np.array_equal(b.observation()[20:].reshape(5,10)[:,6:9],np.zeros((5,3)))
    finally:a.close();b.close()
