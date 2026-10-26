"""图 1：椭圆柱面与坐标系的正投影矢量图。

坐标：X 横向，Y 竖直，Z 纵向；两端为 Z=0 和 Z=L 的平面。
"""
import numpy as np
from matplotlib.patches import Polygon
import sys
sys.dont_write_bytecode = True
from common import plt, save
from style import GEOMETRY, apply_geometry_style, figure_size, pt


def main(output_dir=None):
    apply_geometry_style()
    a, b, L = GEOMETRY['a_m'], GEOMETRY['b_m'], GEOMETRY['length_m']
    alpha, beta = np.deg2rad([GEOMETRY['view_elevation_deg'], GEOMETRY['view_rotation_deg']])
    # Rows are unit and mutually orthogonal: a true orthographic projection.
    P = np.array([
        [-np.cos(beta), 0., np.sin(beta)],
        [np.sin(alpha)*np.sin(beta), np.cos(alpha),
         np.sin(alpha)*np.cos(beta)],
    ])
    assert np.allclose(P @ P.T, np.eye(2))

    def project(xyz):
        return np.asarray(xyz) @ P.T

    def ellipse(t, z=0.):
        t = np.asarray(t)
        return project(np.column_stack([a*np.cos(t), b*np.sin(t), np.full_like(t,z)]))

    fig, ax = plt.subplots(figsize=figure_size('fig01_geometry'))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.90, 7.55)
    ax.set_ylim(-1.65, 1.88)
    ax.axis('off')

    axis_z = project([0.,0.,L])
    dz = axis_z / np.linalg.norm(axis_z)
    normal = np.array([-dz[1], dz[0]])
    e_x, e_y = project([a,0,0]), project([0,b,0])
    t_upper = np.arctan2(normal @ e_y, normal @ e_x)
    upper, lower = ellipse([t_upper, t_upper+np.pi])
    back_visible = ellipse(np.linspace(t_upper,t_upper+np.pi,240),L)
    front_outer = ellipse(np.linspace(t_upper+np.pi,t_upper+2*np.pi,240))
    outline = np.vstack([upper,upper+axis_z,back_visible,lower,front_outer])
    ax.add_patch(Polygon(outline, closed=True, facecolor='white', edgecolor='none',zorder=1))

    # The far end is planar; its hidden half is shown by a fine dashed line.
    back_hidden = ellipse(np.linspace(t_upper+np.pi,t_upper+2*np.pi,240),L)
    ax.plot(*back_hidden.T, color='0.53', lw=.60, dashes=(3.2,2.6),zorder=2)
    ax.plot(*back_visible.T, color='0.18', lw=.85,zorder=3)
    for edge in [upper, lower]:
        ax.plot([edge[0],edge[0]+axis_z[0]], [edge[1],edge[1]+axis_z[1]],
                color='0.18',lw=.85,zorder=3)
    front = ellipse(np.linspace(0,2*np.pi,480))
    ax.add_patch(Polygon(front,closed=True,facecolor='white',edgecolor='0.18',lw=.85,zorder=4))

    def dimension(p, q, offset, label, rotation=0, text_shift=None):
        """Parallel offset dimension between the projections of exact 3-D points."""
        p, q, offset = np.asarray(p), np.asarray(q), np.asarray(offset)
        pp, qq = p+offset, q+offset
        # Extension lines stop short of the physical boundary and extend past arrows.
        u = offset/np.linalg.norm(offset)
        for original, end in [(p,pp),(q,qq)]:
            ax.plot([original[0]+.045*u[0],end[0]+.075*u[0]],
                    [original[1]+.045*u[1],end[1]+.075*u[1]],
                    color='.48',lw=.50,zorder=5)
        ax.annotate('',xy=qq,xytext=pp,
            arrowprops=dict(arrowstyle='<->',lw=.60,color='.20',
                            shrinkA=0,shrinkB=0,mutation_scale=7),zorder=6)
        center=(pp+qq)/2
        if text_shift is not None:
            center+=np.asarray(text_shift)
        ax.text(*center,label,ha='center',va='center',rotation=rotation,
                fontsize=pt(9.5),color='.12',rotation_mode='anchor',
                bbox=dict(boxstyle='square,pad=0.12',fc='white',ec='none'),zorder=7)

    # Longitudinal length on a lower tangent, parallel to Z.
    long_angle=np.rad2deg(np.arctan2(dz[1],dz[0]))
    dimension(lower,lower+axis_z,-.48*normal,rf'$L={L:g}\ \mathrm{{m}}$',
              rotation=long_angle,text_shift=-.13*normal)

    # Transverse diameter: true endpoints X=+/-a in the near end plane.
    transverse_ends=project([[-a,0.,0.],[a,0.,0.]])
    trans_dir=transverse_ends[1]-transverse_ends[0]
    trans_dir/=np.linalg.norm(trans_dir)
    trans_angle=np.rad2deg(np.arctan2(trans_dir[1],trans_dir[0]))-180
    dimension(*transverse_ends,offset=[-.44,-.93],label=rf'$2a={2*a:g}\ \mathrm{{m}}$',
              rotation=trans_angle,text_shift=[-.045,-.28])

    # Vertical diameter at Z=0, Y=+/-b.
    vertical_ends=project([[0.,-b,0.],[0.,b,0.]])
    dimension(*vertical_ends,offset=[-1.38,0.],label=rf'$2b={2*b:g}\ \mathrm{{m}}$',
              rotation=90,text_shift=[-.13,0.])

    # Coordinate triad shares the centre of the near planar end.
    origin=np.array([0.,0.])
    triad = [
        ('X', project([1.,0.,0.]), 1.02, [-.06,.10]),
        ('Y', project([0.,1.,0.]), 1.15, [.0,.12]),
        ('Z', project([0.,0.,1.]), 1.35, [.10,.02]),
    ]
    for label,direction,length,shift in triad:
        endpoint=direction/np.linalg.norm(direction)*length
        ax.annotate('',xy=endpoint,xytext=origin,
            arrowprops=dict(arrowstyle='-|>',lw=.70,color='.18',
                            shrinkA=0,shrinkB=0,mutation_scale=8),zorder=8)
        ax.text(*(endpoint+shift),rf'${label}$',fontsize=pt(10),ha='center',va='center',zorder=9)
    ax.text(-.13,-.11,r'$O$',fontsize=pt(9),ha='center',va='center',zorder=9)

    return save(fig, 'fig01_geometry', output_dir, dict(
        source='Orthographic vector drawing; no raster source',
        geometry=GEOMETRY,
        projection_matrix=P.tolist(),
        coordinate_system={'X': 'transverse', 'Y': 'vertical', 'Z': 'longitudinal'},
    ))


if __name__ == '__main__':
    result = main()
    print('已生成：', result['stem'])
