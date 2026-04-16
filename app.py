import streamlit as st
import matplotlib.pyplot as plt
import physics_engine as pe

# 图表字体与渲染配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="小孔成像仿真", layout="wide")
st.title("小孔成像仿真系统")

# 参数设置
with st.sidebar:
    st.subheader("光学系统参数")
    source_type = st.selectbox("光源形态", ["非对称字母 F", "日偏食 (月牙)"])
    aperture_type = st.selectbox("孔径形态", ["圆形", "正方形", "树叶缝隙 (不规则)"])
    
    st.markdown("---")
    d1 = st.slider("物距 (d1)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    d2 = st.slider("像距 (d2)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
    aperture_size = st.slider("孔径尺寸系数", min_value=1, max_value=100, value=5, step=1)

# 数据计算缓存
@st.cache_data
def get_cached_matrices(src_type, apt_type, apt_size):
    src = pe.create_source(src_type)
    apt = pe.create_aperture(apt_type, apt_size)
    return src, apt

src_matrix, apt_matrix = get_cached_matrices(source_type, aperture_type, aperture_size)
img_matrix = pe.compute_image_2d(src_matrix, apt_matrix)

# 三维几何光路视图
st.subheader("三维几何光路")
fig_3d = pe.create_3d_ray_diagram(source_type, d1, d2, aperture_type, aperture_size)
st.plotly_chart(fig_3d, use_container_width=True)

# 二维像面光强分布视图
st.subheader("二维像面光强分布")
col1, col2, col3 = st.columns(3)

def plot_matrix(matrix, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.imshow(matrix, cmap='inferno', origin='upper')
    ax.axis('off')
    ax.set_title(title, color='white', pad=10)
    return fig

with col1:
    st.pyplot(plot_matrix(src_matrix, "光源 S(x,y)"))
with col2:
    st.pyplot(plot_matrix(apt_matrix, "孔径透过率 A(x,y)"))
with col3:
    st.pyplot(plot_matrix(img_matrix, "像面光强 I(x,y)"))

# 系统状态输出
st.markdown("---")
st.markdown(f"**系统当前状态**: 放大倍率 $M = d_2/d_1 = {d2/d1:.2f}$")