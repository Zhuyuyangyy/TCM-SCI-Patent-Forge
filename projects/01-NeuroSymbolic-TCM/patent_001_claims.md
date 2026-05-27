# Target 001 专利权利要求书 — 高光谱NeRF舌象重建

## 独立权利要求

**权利要求1.** 一种基于三维神经辐射场的高光谱舌组织结构及血氧定量重建方法，其特征在于，包括以下步骤：

S1. 采集舌体在不少于32个视角下的高光谱图像序列，所述高光谱图像的工作波段范围为500nm至950nm，光谱分辨率优于5nm，空间分辨率优于10μm；

S2. 构建扩展型神经辐射场网络，将标准NeRF的RGB三通道输出扩展为N通道光谱辐射输出，其中N≥120；所述网络以三维空间坐标(x,y,z)和观察方向(θ,φ)为输入，输出该采样点的体积密度σ和N通道光谱辐射值L(λ₁,...,λₙ)；

S3. 在所述神经辐射场网络的特征提取层增设血氧饱和度预测分支，利用氧合血红蛋白与脱氧血红蛋白在800nm等吸收点及560nm、760nm特征吸收峰的物理特性，通过改进比率法计算各体素的血氧饱和度SpO₂值；

S4. 设计复合损失函数L = L_spec + λ₁·L_SpO₂ + λ₂·L_vessel + λ₃·L_eik，其中L_spec为光谱重建损失，L_SpO₂为物理约束的血氧一致性损失，L_vessel为血管分割损失，L_eik为密度场Eikonal正则化损失；

S5. 通过体渲染积分获取任意视角下的高光谱渲染图像，并从训练完成的辐射场中提取三维血管网络拓扑结构。

## 从属权利要求

**权利要求2.** 根据权利要求1所述的方法，其特征在于，所述步骤S2中的扩展型神经辐射场网络包含8层全连接层，每层256个神经元，采用ReLU激活函数，在第4层设置跳跃连接(skip connection)以增强梯度传播；所述网络的光谱辐射输出层采用Sigmoid激活函数将输出归一化至[0,1]区间。

**权利要求3.** 根据权利要求1所述的方法，其特征在于，所述步骤S3中的血氧饱和度计算采用公式SpO₂ = a₀ + a₁·R₁ + a₂·R₂ + a₃·R₁·R₂，其中R₁为760nm与800nm处反射率的比值，R₂为560nm与580nm处反射率的比值，系数a₀至a₃通过临床标定确定。

**权利要求4.** 根据权利要求1所述的方法，其特征在于，所述步骤S5中的三维血管网络提取包括：在辐射场的三维体素空间中，将SpO₂值低于阈值且体积密度高于阈值的体素标记为候选血管体素；采用三维连通分量分析提取血管拓扑结构；通过血管横截面密度分布估算血管直径，通过血氧饱和度梯度估算血流速度。

**权利要求5.** 一种基于三维神经辐射场的高光谱舌组织结构及血氧定量重建系统，其特征在于，包括：高光谱图像采集模块、多视角图像配准模块、扩展型神经辐射场训练模块、血氧饱和度物理约束模块、三维血管网络提取模块和舌下络脉血流动力学参数映射模块。

---

## 创新防御论证 (Agent-Innovation)

### 为何不用纯2D舌诊？
| 问题 | 2D方案缺陷 | NeRF方案优势 |
|------|-----------|-------------|
| 环境光干扰 | 依赖色温校正，仍有残差 | 3D隐式表示分离光照与几何 |
| 舌面反光 | 高光区域信息丢失 | 多视角融合消除镜面反射 |
| 几何形变 | 2D配准误差大 | 3D空间刚性配准 |
| 微循环不可见 | 仅获取表层信息 | 体积渲染穿透表层 |
| 血氧无法量化 | RGB通道无光谱信息 | 120通道高光谱+物理模型 |

### 为何不用纯深度学习？
- 纯数据驱动的SpO₂预测缺乏物理可解释性
- 本方案将Beer-Lambert物理定律嵌入损失函数，确保预测结果符合血液光学物理规律
- Eikonal正则化保证密度场几何一致性

---

## SCI论文数学形式化 (Agent-Paper)

### Volume Rendering Equation (Hyperspectral Extension)

Given a ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ with origin $\mathbf{o}$ and direction $\mathbf{d}$, the rendered spectral radiance at wavelength $\lambda_k$ is:

$$\hat{L}(\lambda_k) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot L(\mathbf{r}(t), \mathbf{d}, \lambda_k) \, dt$$

where $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$ is the accumulated transmittance.

### Composite Loss Function

$$\mathcal{L} = \underbrace{\sum_{k=1}^{N} \left\| \hat{L}(\lambda_k) - L^{gt}(\lambda_k) \right\|^2}_{\mathcal{L}_{spec}} + \lambda_1 \underbrace{\left\| \hat{S}_{O_2} - S_{O_2}^{phys} \right\|^2}_{\mathcal{L}_{SpO_2}} + \lambda_2 \underbrace{\text{BCE}(\hat{V}, V^{gt})}_{\mathcal{L}_{vessel}} + \lambda_3 \underbrace{\mathbb{E}\left[ \left( \| \nabla \sigma \|_2 - 1 \right)^2 \right]}_{\mathcal{L}_{eik}}$$

where $S_{O_2}^{phys}$ is the physics-based SpO₂ estimate from the modified Beer-Lambert law:

$$S_{O_2} = a_0 + a_1 \frac{R_{760}}{R_{800}} + a_2 \frac{R_{560}}{R_{580}} + a_3 \frac{R_{760}}{R_{800}} \cdot \frac{R_{560}}{R_{580}}$$
