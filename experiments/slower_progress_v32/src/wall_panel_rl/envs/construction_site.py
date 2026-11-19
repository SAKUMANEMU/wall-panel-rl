"""Small-scale construction mock-up with persistent physical floor/ceiling.

The ceiling collider is continuous; its visual cutaway is only for visibility.
Robot-base/floor contact is the intentional bolted mounting interface. Moving
robot links, the panel, the wall and all dynamic loads retain physical collisions.
"""
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation


def wall_geometry(configuration, original_center, orientation):
    site=configuration.get('construction_site',{})
    if not site.get('enabled'):
        return [.8,.8,configuration['target']['wall_thickness']],original_center
    vertical=Rotation.from_quat(orientation).apply([1.,0.,0.])
    if abs(vertical[2])<.9:
        raise ValueError('Construction installation wall must be approximately vertical')
    height=site['ceiling_z']-site['floor_z']
    middle=(site['ceiling_z']+site['floor_z'])/2
    # Translate within the wall plane so the installation clearance is preserved.
    center=np.asarray(original_center)+vertical*((middle-original_center[2])/vertical[2])
    return [height/abs(vertical[2]),site['wall_width'],configuration['target']['wall_thickness']],center


def build_site(env):
    env.site_collision_ids=[]
    env.site_visual_ids=[]
    env.floor=None
    env.ceiling=None
    env.load_cable_ids=[]
    site=env.config.get('construction_site',{})
    if not site.get('enabled'):
        return
    floor=float(site['floor_z']);ceiling=float(site['ceiling_z']);thickness=float(site['slab_thickness'])
    width,depth=map(float,site['footprint'])
    if not 0 <= floor < ceiling or min(thickness,width,depth)<=0:
        raise ValueError('Positive site dimensions and ordered floor/ceiling heights required')
    # A 4 m footprint covers the complete reach of this fixed-base robot.
    env.floor=env.box([width,depth,thickness],[0,0,floor-thickness/2],[.48,.47,.44,1])
    env.ceiling=env.box([width,depth,thickness],[0,0,ceiling+thickness/2],[.7,.7,.69,1],
                        visual=not site.get('cutaway_ceiling',True))
    env.site_collision_ids=[env.floor,env.ceiling]
    for body in env.site_collision_ids:
        p.changeDynamics(body,-1,lateralFriction=.85,restitution=0.,physicsClientId=env.client)
    p.setCollisionFilterPair(env.robot,env.floor,-1,-1,0,physicsClientId=env.client)
    # Visibility flags never remove physical fixtures from training or queries.
    pallet=env.box([.55,.36,.08],[-1.35,1.45,floor+.04],[.64,.43,.22,1],
                   visual=site.get('visual_details',True))
    boards=env.box([.48,.30,.13],[-1.35,1.45,floor+.145],[.78,.75,.65,1],
                   visual=site.get('visual_details',True))
    env.site_collision_ids.extend([pallet,boards])
    if not site.get('visual_details',True):
        return
    def visual(dimensions,position,color):
        body=env.box(dimensions,position,color,collision=False)
        env.site_visual_ids.append(body)
        return body
    concrete=[.61,.60,.56,1];steel=[.23,.27,.29,1];timber=[.64,.43,.22,1]
    # Concrete construction joints and a bolted robot mounting plate.
    for x in np.arange(-1.6,1.61,.4):
        visual([.007,depth,.001],[x,0,floor+.0006],[.34,.34,.32,1])
    for y in np.arange(-1.6,1.61,.4):
        visual([width,.007,.001],[0,y,floor+.0006],[.34,.34,.32,1])
    visual([.34,.30,.018],[0,0,floor+.009],steel)
    for x in (-.13,.13):
        for y in (-.11,.11):visual([.022,.022,.012],[x,y,floor+.024],[.14,.15,.16,1])
    # Hazard striping marks the reachable installation work bay.
    for x in np.arange(-1.35,.91,.10):
        color=[.92,.68,.08,1] if int(round(x*10))%2 else [.12,.13,.13,1]
        visual([.095,.065,.002],[x,-1.17,floor+.002],color)
    # Back-side framing is behind the physical installation wall. It cannot be
    # mistaken for an extra observed moving obstacle.
    wall_center,wall_quat=p.getBasePositionAndOrientation(env.wall,physicsClientId=env.client)
    normal=Rotation.from_quat(wall_quat).apply([0,0,1])
    wall_x=wall_center[0]+normal[0]*env.config['target']['wall_thickness']/2+.025
    for y in np.arange(-1.6,1.61,.4):
        visual([.055,.055,ceiling-floor],[wall_x,y,(floor+ceiling)/2],timber)
    for z in (floor+.025,ceiling-.025):
        visual([.065,site['wall_width'],.05],[wall_x,0,z],timber)
    # Exposed ceiling joists show the overhead limit while the continuous roof
    # collision slab stays invisible for a readable camera view.
    cutaway=site.get('cutaway_ceiling',True)
    for y in ((1.55,) if cutaway else (-1.55,1.55)):
        visual([width,.07,.09],[0,y,ceiling+.045],steel)
    for x in (-1.5,-.8,-.1,.6):
        visual([.045,.9 if cutaway else 3.1,.065],[x,1.10 if cutaway else 0,ceiling+.045],concrete)
    visual([width,.55,thickness],[0,1.55,ceiling+thickness/2],concrete)
    # Stored boards and materials are outside the dynamic work bay; keep their
    # small static footprint physical and include it in all collision queries.
    for y in (-.15,.15):
        visual([.018,.018,.36],[wall_center[0]-.03,y,float(env.target_pose[0][2])],[.85,.68,.12,1])
    if env.render_mode=='human':
        p.addUserDebugText(f'CEILING z={ceiling:.2f} m (cutaway)',[-1.25,1.55,ceiling+.08],
                          [0.22,.24,.24],textSize=1.,physicsClientId=env.client)
        p.addUserDebugText(f'FLOOR z={floor:.2f} m',[-1.1,-1.18,floor+.025],
                          [.9,.7,.1],textSize=1.,physicsClientId=env.client)


def create_load(env,dimensions,center,index):
    """Visual construction material bundle with the exact original box collider."""
    colors=([.64,.31,.20,1],[.22,.38,.48,1],[.68,.62,.47,1],[.43,.47,.48,1],[.46,.34,.23,1])
    half=np.asarray(dimensions)/2
    shapes=[p.GEOM_BOX]*3
    halves=[half.tolist(),[half[0]+.0008,.008,half[2]+.0008],
            [half[0]+.0008,.008,half[2]+.0008]]
    positions=[[0,0,0],[0,-half[1]*.55,0],[0,half[1]*.55,0]]
    shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=half.tolist(),physicsClientId=env.client)
    appearance=p.createVisualShapeArray(shapeTypes=shapes,halfExtents=halves,
        visualFramePositions=positions,rgbaColors=[colors[index%len(colors)],[.16,.18,.19,1],[.16,.18,.19,1]],physicsClientId=env.client)
    return p.createMultiBody(0,shape,appearance,center,[0,0,0,1],physicsClientId=env.client)


def update_site_visuals(env):
    """Illustrative overhead-hoist cables, updated only for interactive display.

    Motion remains prescribed; cables are not a simulated pendulum or safety
    constraint. Replacing existing lines avoids a growing trail during replay.
    """
    site=env.config.get('construction_site',{})
    if env.render_mode!='human' or not site.get('enabled'):
        return
    for slot,(center,dimensions) in enumerate(env.obstacles):
        old=env.load_cable_ids[slot] if slot<len(env.load_cable_ids) else -1
        cable=p.addUserDebugLine([center[0],center[1],site['ceiling_z']],
            [center[0],center[1],center[2]+dimensions[2]/2],[.19,.20,.20],2.,
            replaceItemUniqueId=old,physicsClientId=env.client)
        if slot<len(env.load_cable_ids):env.load_cable_ids[slot]=cable
        else:env.load_cable_ids.append(cable)
