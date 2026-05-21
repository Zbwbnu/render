import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import pytorch3d
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import ico_sphere
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer, \
    SoftSilhouetteShader, BlendParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"PyTorch3D 版本: {pytorch3d.__version__}")

model_file = "cow.obj"
if not os.path.isfile(model_file):
    raise FileNotFoundError("错误：请确保 cow.obj 在当前目录下")

vertices, faces_data, _ = load_obj(model_file)
faces = faces_data.verts_idx.to(device)
vertices = vertices.to(device)

center_verts = vertices - vertices.mean(dim=0)
scale = torch.max(torch.abs(center_verts))
normal_verts = center_verts / scale

target_mesh = Meshes(verts=[normal_verts], faces=[faces])

view_count = 20
distance = 2.7
elevation_list = torch.zeros(view_count)
azimuth_list = torch.linspace(-180, 180, view_count)

R, T = look_at_view_transform(distance, elevation_list, azimuth_list)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1. / 1e-4 - 1.) * 1e-4,
    faces_per_pixel=50
)

silhouette_shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

with torch.no_grad():
    target_mesh_batch = target_mesh.extend(view_count)
    render_result = rasterizer(target_mesh_batch)
    gt_silhouette = silhouette_shader(render_result, target_mesh_batch)[..., 3]

sphere_mesh = ico_sphere(4, device)
offset = torch.zeros_like(sphere_mesh.verts_packed(), requires_grad=True)
optimizer = torch.optim.SGD([offset], lr=1.2, momentum=0.92)

out_dir = "mesh_outputs"
os.makedirs(out_dir, exist_ok=True)
print(f"模型保存路径: {out_dir}")

total_epochs = 300
for epoch in range(total_epochs):
    optimizer.zero_grad()
    deformed_mesh = sphere_mesh.offset_verts(offset)
    deformed_batch = deformed_mesh.extend(view_count)
    pred_render = rasterizer(deformed_batch)
    pred_sil = silhouette_shader(pred_render, deformed_batch)[..., 3]

    loss_sil = torch.mean((pred_sil - gt_silhouette) ** 2)
    loss_lap = mesh_laplacian_smoothing(deformed_mesh)
    loss_edge = mesh_edge_loss(deformed_mesh)
    loss_normal = mesh_normal_consistency(deformed_mesh)

    total_loss = loss_sil + 1.0 * loss_lap + 0.12 * loss_edge + 0.015 * loss_normal
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == total_epochs - 1:
        clear_output(wait=True)
        print(f"迭代: {epoch:03d}/{total_epochs} | 总损失: {total_loss:.4f} | 剪影误差: {loss_sil:.4f}")
        v = deformed_mesh.verts_list()[0]
        f = deformed_mesh.faces_list()[0]
        save_path = os.path.join(out_dir, f"deform_epoch_{epoch:03d}.obj")
        save_obj(save_path, v, f)
        print(f"已保存: {save_path}")
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_silhouette[0].cpu().numpy(), cmap="gray")
        plt.title("Target Silhouette")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(pred_sil[0].detach().cpu().numpy(), cmap="gray")
        plt.title(f"Current Epoch {epoch}")
        plt.axis("off")
        plt.show()

print("优化完成！模型已保存至 mesh_outputs 文件夹")