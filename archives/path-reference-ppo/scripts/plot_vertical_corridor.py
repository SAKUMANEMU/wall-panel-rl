"""Plot measured passage events and the whole panel's vertical bounds."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wall_panel_rl.utils.config import ROOT
from probe_vertical_corridor import passage_measurements


def main():
    root=ROOT/'results/diagnostics/four_vertical_corridor_v19'
    payload=json.loads((ROOT/'results/replays/vertical_corridor_1166026.json').read_text())
    data=passage_measurements(payload)
    (root/'verified_passages.json').write_text(json.dumps(data,indent=2)+'\n')
    frames=payload['trajectory'];scene=payload['case']['scene']
    xyz=np.array([f['panel'] for f in frames]);times=np.array([f['time_s'] for f in frames])
    half_z=np.abs(Rotation.from_quat([f['panel_orientation'] for f in frames]).as_matrix())[:,2,:] @ (np.array(payload['configuration']['panel']['dimensions'])/2)
    fig=plt.figure(figsize=(14.5,6.3),layout='constrained')
    ax=fig.add_subplot(121,projection='3d')
    ax.plot(*xyz.T,color='royalblue',lw=2,label='Actual panel-center trajectory')
    route=np.array(scene['nominal_motion']['panel_positions'])
    ax.plot(*route.T,':',color='gray',alpha=.7,label='Preliminary placement route')
    for i,box in enumerate(scene['obstacles']):
        centers=np.array([f['obstacle_centers'][i] for f in frames])
        j=next((p['frame'] for p in data['ordered_alternating_passages'] if p['slot']==i),int(np.argmin(np.linalg.norm((xyz-centers)[:,:2],axis=1))))
        center=centers[j];dims=np.array(box['dimensions'])
        ax.plot(*centers.T,color='darkorange',alpha=.35,lw=.8)
        for axis in range(3):
            others=[k for k in range(3) if k!=axis]
            for a in [-1,1]:
                for b in [-1,1]:
                    corners=np.stack([center-dims/2,center+dims/2])
                    corners[:,others[0]]=center[others[0]]+a*dims[others[0]]/2
                    corners[:,others[1]]=center[others[1]]+b*dims[others[1]]/2
                    ax.plot(*corners.T,color='darkorange',alpha=.7,lw=1.)
        ax.text(*center,f' O{i+1}',fontsize=8)
    for i,event in enumerate(data['ordered_alternating_passages']):
        point=xyz[event['frame']]
        ax.scatter(*point,color='navy',s=25)
        ax.text(*point,f'  {i+1}: O{event["slot"]+1} {event["side"]}',fontsize=8,color='navy')
    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Boxes shown at each nearest/passage encounter')
    ax.legend(loc='upper left',fontsize=8)
    ax=fig.add_subplot(122)
    ax.fill_between(times,xyz[:,2]-half_z,xyz[:,2]+half_z,color='royalblue',alpha=.12,label='Whole panel vertical bounds')
    ax.plot(times,xyz[:,2],color='royalblue',lw=2,label='Panel center')
    colors=['#e18b3a','#36a16b','#9d6abd','#c85757']
    for i,box in enumerate(scene['obstacles']):
        centers=np.array([f['obstacle_centers'][i] for f in frames]);half=box['dimensions'][2]/2
        ax.plot(times,centers[:,2],color=colors[i],lw=1,label=f'O{i+1} center')
        ax.fill_between(times,centers[:,2]-half,centers[:,2]+half,color=colors[i],alpha=.07)
    for i,event in enumerate(data['ordered_alternating_passages']):
        ax.axvline(event['time_s'],color='navy',ls=':',lw=.8)
        ax.scatter(event['time_s'],event['panel_z'],color='navy',s=25)
        ax.annotate(f'{i+1}: O{event["slot"]+1} {event["side"]}',
                    (event['time_s'],event['panel_z']),xytext=(-10,15 if i!=1 else -30),
                    textcoords='offset points',ha='right',fontsize=8,color='navy')
    ax.set(xlabel='Simulation time (s)',ylabel='Z (m)',title='Measured rise and fall, with full-panel clearance')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8,loc='lower left',ncol=2)
    result=payload['result']
    fig.suptitle(f'Larger vertical stagger — seed {result["seed"]}, {result["reason"]} in {result["steps"]} real steps',fontsize=14)
    fig.supxlabel('Selected development demonstration; no training. Shaded bounds and box poses are measured physical states.',fontsize=9)
    fig.savefig(root/'trajectory.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
