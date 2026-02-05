# Installation

```
python -m pip install --no-cache-dir torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-cache-dir torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
git clone https://github.com/nv-tlabs/NKSR.git
cd NKSR
pip install "python-pycg[all]"
pip install randomname pykdtree plyfile flatten-dict pyntcloud
export MAX_JOBS=16
pip install --no-build-isolation -v package/
```
refer to issue of videomimic: https://github.com/hongsukchoi/VideoMimic/issues/9
download prebuilt nksr from https://nksr.s3.ap-northeast-1.amazonaws.com/whl/torch-2.0.0%2Bcu118.html
download ks.pt from https://drive.google.com/file/d/11iPBBJ4Tj4mUaHGnhSr_uWstQxWU3fPj/view

# Running Conversion
'''
 python -m omniretargeting.workflows.her.scene.ply2scene.convert \                       
  --seq smooth --robot g1
'''

'''
python -m omniretargeting.workflows.her.scene.ply2scene.viser_preview \                 
  --seq smooth --robot g1 \
  --scene_obj /home/juyiang/data/holosoma_runs/smooth/g1/artifacts/ply2scene/scene/meshes/scene_visual.obj \
  --scene_xml /home/juyiang/data/holosoma_runs/smooth/g1/artifacts/ply2scene/scene/scene.xml \
  --show-points
'''

# ply2scene

`ply2scene` 用于把 Holosoma 的 RGBD 重建结果拼接成场景点云，并重建出可用于仿真/可视化的静态场景网格，最终导出：
- `meshes/scene_visual.obj`：渲染用网格
- `meshes/scene_collision.obj`：碰撞用网格（当前与 visual 相同）
- `scene.urdf`：yourdfpy/Viser 可加载
- `scene.xml`：MuJoCo MJCF 可加载
- `meta.json`：运行参数与统计

本目录包含两个脚本：
- `convert.py`：生成场景包
- `viser_preview.py`：在 Viser 中预览（点云 vs OBJ vs MJCF 网格）

## 依赖与环境

在仓库根目录下运行（示例路径按你的机器）：

```bash
cd /home/juyiang/data/omniretargeting
conda activate hsretargeting
export PYTHONPATH=/home/juyiang/data/omniretargeting
```

- `convert.py` 需要：`open3d`, `opencv-python`, `numpy`, `tyro`（以及可选 `nksr`, `torch`）
- `viser_preview.py` 需要：`viser`（可选 `mujoco` 用于加载 `scene.xml`）

## 1) 生成场景（convert）

最小用法：

```bash
python -m omniretargeting.workflows.her.scene.ply2scene.convert \
  --seq <SEQ_NAME> \
  --robot g1
```

默认输出目录：`<run_dir>/artifacts/ply2scene/scene/`。

### 常用参数

- **网格后端**：`--mesh-method {auto,poisson,nksr}`
  - `auto`：优先 NKSR（如果可 import），否则 Poisson
  - `nksr`：细节更好，依赖 `torch+nksr`
  - `poisson`：更稳但更“糊”，易外扩，需要密度过滤/裁剪

- **NKSR checkpoint**：`--nksr-checkpoint-path <path>`
  - 默认是 `None`，可显式指定本地 checkpoint
  - 如果换机器路径不存在，直接在命令行覆盖或设为 `null`（让 NKSR 走默认下载/加载逻辑）。

- **点云质量/密度（影响很大）**
  - `--voxel-size`：体素采样尺度（更小更细但更慢更吃内存），常见 `0.1 → 0.05`
  - `--max-points`：全局点数上限（更大更细但更慢）
  - `--voxel-max-points-per-cell`：每个 voxel 最多保留点数
  - `--depth-gradient-threshold-m`：深度梯度过滤阈值（越小越干净但容易缺点）

- **法线估计（影响 NKSR/Poisson）**
  - `--normal-radius`、`--normal-max-nn`
  - `--orient-normals-k`：法线一致性约束（Poisson 对此非常敏感）

- **补洞（当前是“顶视补洞”，主要补地面/水平面）**
  - `--hole-fill-resolution`：XY 网格分辨率（越大补得越细但更慢）
  - `--hole-fill-knn` / `--hole-fill-power`：IDW 插值 kNN 与幂指数
  - `--hole-fill-max-points`：补洞点上限（控制二次重建耗时/显存）

- **裁剪**：`--crop-to-aabb/--no-crop-to-aabb`
  - 开启后把网格裁剪到点云 AABB，减少“飞面/外扩”；但点云本身缺失时也可能把边缘切掉。

- **输出**
  - `--output-dir`：自定义输出目录
  - `--write-includes`：额外写 `scene_assets.xml` / `scene_body.xml`（mujocoinclude 形式）

## 2) 预览检查（viser_preview）

`viser_preview.py` 用于对齐检查：
- （可选）显示从 RGBD 拼接出的点云
- 显示 `scene_visual.obj`
- （可选）从 `scene.xml` 反解 MuJoCo world mesh 并显示

示例：

```bash
python -m holosoma_retargeting.ply2scene.viser_preview \
  --seq <SEQ_NAME> --robot g1 \
  --show-points \
  --scene-obj <scene_dir>/meshes/scene_visual.obj \
  --scene-xml <scene_dir>/scene.xml
```

常用参数：
- `--show-points`：显示点云（默认关闭）
- `--scene-source {rgbd,fused_ply,aligned_ply,ply}`：点云来源
  - `rgbd`：从 RGBD 拼接（原默认行为）
  - `fused_ply`：使用 `fused_scene.ply`（会应用对齐矩阵）
  - `aligned_ply`：使用 `aligned_scene_manual_<seq>.ply`（已对齐，不再重复对齐）
  - `ply`：使用 `--scene-ply <path>` 指定的 PLY
- `--scene-ply`：当 `--scene-source ply` 时提供
- `--max-points`：预览点云下采样
- `--apply-scale-factor/--no-apply-scale-factor`：是否应用 `pipeline_config_json` 里的 `scale_factor`
- `--point-size` / `--mesh-opacity`

运行后终端会打印 Viser URL，浏览器打开即可。

## 3) 输出目录结构

`convert.py` 输出目录形如：

```text
<scene_out_dir>/
  meshes/
    scene_visual.obj
    scene_collision.obj
  scene.urdf
  scene.xml
  meta.json
  scene_assets.xml        # 可选（--write-includes）
  scene_body.xml          # 可选（--write-includes）
```

## 4) 常见问题

- **墙面/垂直方向有孔洞，但地板没洞**
  - 当前补洞是“顶视高度图插值”，天然更擅长补地面，不擅长补墙。
  - 优先提升墙面点云密度：减小 `--voxel-size`，提高 `--max-points` / `--voxel-max-points-per-cell`，并适当放松 `--depth-gradient-threshold-m`。
  - 其次再考虑降低 `--nksr-detail-level`（更细更慢）。

- **Poisson 外扩/飞面多**
  - 保持 `--crop-to-aabb`，并适当提高 `--poisson-density-quantile`（更狠地丢低密度区域）。
