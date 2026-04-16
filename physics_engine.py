import numpy as np
from scipy.signal import fftconvolve
import plotly.graph_objects as go

def create_source(shape_type, resolution=512):
    """生成光源二维光强分布矩阵"""
    img = np.zeros((resolution, resolution))
    if shape_type == "非对称字母 F":
        img[100:400, 150:200] = 1.0
        img[100:150, 200:350] = 1.0
        img[220:270, 200:300] = 1.0
    elif shape_type == "日偏食 (月牙)":
        y, x = np.ogrid[-resolution/2:resolution/2, -resolution/2:resolution/2]
        mask1 = x**2 + y**2 <= (resolution/4)**2
        mask2 = (x - resolution/10)**2 + (y - resolution/10)**2 <= (resolution/4)**2
        img[mask1 & ~mask2] = 1.0
    return img

def create_aperture(shape_type, size_factor, resolution=512):
    """生成孔径二维透过率矩阵"""
    img = np.zeros((resolution, resolution))
    r = max(1, int((resolution / 2) * (size_factor / 100.0) * 0.5)) 
    y, x = np.ogrid[-resolution/2:resolution/2, -resolution/2:resolution/2]
    
    if shape_type == "圆形":
        mask = x**2 + y**2 <= r**2
        img[mask] = 1.0
    elif shape_type == "正方形":
        img[resolution//2 - r : resolution//2 + r, resolution//2 - r : resolution//2 + r] = 1.0
    elif shape_type == "树叶缝隙 (不规则)":
        mask_circle = x**2 + y**2 <= r**2
        mask_triangle = y > -x + r/2
        mask_triangle2 = y < x + r
        img[mask_circle & mask_triangle & mask_triangle2] = 1.0
        
    total_area = np.sum(img)
    if total_area > 0:
        img = img / total_area
    return img

def compute_image_2d(source, aperture):
    """计算像面光强分布 (二维卷积)"""
    result = fftconvolve(source, aperture, mode='same')
    return np.rot90(result, 2)

def create_3d_ray_diagram(src_type, d1, d2, apt_type, apt_size):
    """构建三维几何光路与光斑叠加效果图"""
    fig = go.Figure()

    # 离散化物面骨架点
    if src_type == "非对称字母 F":
        src_pts = [[-0.4, y] for y in np.linspace(-0.8, 0.8, 8)] + \
                  [[x, 0.8] for x in np.linspace(-0.4, 0.4, 4)] + \
                  [[x, 0.1] for x in np.linspace(-0.4, 0.2, 3)]
    else:
        src_pts = [[0.6*np.cos(t), 0.6*np.sin(t)] for t in np.linspace(0, 2*np.pi, 15) if not (0 < t < np.pi/2)]

    r = max(0.015, (apt_size / 100.0) * 0.4)
    if apt_type == "圆形":
        hole_v = [[r*np.cos(t), r*np.sin(t)] for t in np.linspace(0, 2*np.pi, 16)]
    elif apt_type == "正方形":
        hole_v = [[-r, -r], [r, -r], [r, r], [-r, r]]
    else:
        hole_v = [[0, r], [r*0.866, -r*0.5], [-r*0.866, -r*0.5]]

    # 绘制参考平面
    def add_plane(x_pos, color, name, size):
        fig.add_trace(go.Mesh3d(
            x=[x_pos]*4, y=[-size, size, size, -size], z=[-size, -size, size, size],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=0.08, name=name, showscale=False, hoverinfo='skip'
        ))

    add_plane(-d1, '#00FFFF', '物面', 1.5)
    add_plane(0, '#FFFFFF', '光阑面', 1.5)
    add_plane(d2, '#0055FF', '像面', 1.5 * max(1, d2/d1))

    # 绘制物面光源实体点
    src_y = [p[0] for p in src_pts]
    src_z = [p[1] for p in src_pts]
    fig.add_trace(go.Scatter3d(
        x=[-d1]*len(src_pts), y=src_y, z=src_z,
        mode='markers', marker=dict(size=4, color='#00FFFF', symbol='circle'), 
        showlegend=False, hoverinfo='skip'
    ))

    # 光斑投影计算矩阵
    kH = (d1 + d2) / d1
    cSc = -d2 / d1

    mesh_x, mesh_y, mesh_z = [], [], []
    mesh_i, mesh_j, mesh_k = [], [], []
    v_idx = 0

    for p in src_pts:
        px, py = p[0], p[1]
        for hv in hole_v:
            mesh_x.append(d2)
            mesh_y.append(hv[0]*kH + px*cSc)
            mesh_z.append(hv[1]*kH + py*cSc)

        num_v = len(hole_v)
        for v in range(1, num_v - 1):
            mesh_i.append(v_idx)
            mesh_j.append(v_idx + v)
            mesh_k.append(v_idx + v + 1)
        v_idx += num_v

    if mesh_x:
        fig.add_trace(go.Mesh3d(
            x=mesh_x, y=mesh_y, z=mesh_z,
            i=mesh_i, j=mesh_j, k=mesh_k,
            color='#FF9900', opacity=0.35, showscale=False, hoverinfo='skip'
        ))

    # 绘制光阑面孔径轮廓线
    hx, hy = zip(*(hole_v + [hole_v[0]]))
    fig.add_trace(go.Scatter3d(
        x=[0]*len(hx), y=hx, z=hy,
        mode='lines', line=dict(color='#FFFFFF', width=4), showlegend=False
    ))

    # 绘制边缘参考射线
    for p in [src_pts[0], src_pts[-1]]:
        px, py = p[0], p[1]
        hv = hole_v[0]
        ix = hv[0]*kH + px*cSc
        iy = hv[1]*kH + py*cSc
        fig.add_trace(go.Scatter3d(
            x=[-d1, 0, d2], y=[px, hv[0], ix], z=[py, hv[1], iy],
            mode='lines', line=dict(color='rgba(0, 255, 150, 0.7)', width=2), showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Z (光轴)', yaxis_title='X', zaxis_title='Y',
            xaxis=dict(range=[-5.5, 5.5], showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(range=[-3, 3], showgrid=False, zeroline=False, showbackground=False),
            zaxis=dict(range=[-3, 3], showgrid=False, zeroline=False, showbackground=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        template="plotly_dark",
        height=500
    )
    return fig