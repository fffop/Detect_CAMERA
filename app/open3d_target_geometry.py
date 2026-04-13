from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from app.realsense_rgbd import CameraIntrinsics


@dataclass(frozen=True)
class Open3DGeometryConfig:
    min_depth_m: float = 0.10
    max_depth_m: float = 1.50
    voxel_size_m: float = 0.0025
    min_point_count: int = 120
    outlier_nb_neighbors: int = 24
    outlier_std_ratio: float = 1.5
    dbscan_eps_m: float = 0.006
    dbscan_min_points: int = 20
    axis_scale_m: float = 0.03

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TargetGeometry:
    center_xyz_m: list[float]
    rotation_matrix: list[list[float]]
    euler_xyz_deg: list[float]
    quaternion_xyzw: list[float]
    extent_xyz_m: list[float]
    point_count: int
    obb_corners_xyz_m: list[list[float]]
    status: str = "ok"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TargetGeometryEstimate:
    status: str
    mask_pixel_count: int
    valid_depth_pixel_count: int
    raw_point_count: int
    filtered_point_count: int
    clustered_point_count: int
    geometry: TargetGeometry | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["geometry"] = None if self.geometry is None else self.geometry.to_dict()
        return payload


def estimate_target_geometry(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray,
    config: Open3DGeometryConfig,
) -> TargetGeometryEstimate:
    mask_bool = np.asarray(mask, dtype=bool)
    if depth_m.shape[:2] != mask_bool.shape[:2]:
        raise ValueError("depth and mask must have the same image size")

    masked_depth = np.asarray(depth_m, dtype=np.float32).copy()
    mask_pixel_count = int(mask_bool.sum())
    valid = (
        mask_bool
        & np.isfinite(masked_depth)
        & (masked_depth >= float(config.min_depth_m))
        & (masked_depth <= float(config.max_depth_m))
    )
    valid_depth_pixel_count = int(valid.sum())
    if valid_depth_pixel_count < max(16, int(config.min_point_count // 4)):
        return TargetGeometryEstimate(
            status="insufficient_depth",
            mask_pixel_count=mask_pixel_count,
            valid_depth_pixel_count=valid_depth_pixel_count,
            raw_point_count=0,
            filtered_point_count=0,
            clustered_point_count=0,
        )
    masked_depth[~valid] = 0.0

    pcd = _point_cloud_from_masked_depth(masked_depth, intrinsics, config.max_depth_m)
    raw_point_count = int(len(pcd.points))
    if raw_point_count == 0:
        return TargetGeometryEstimate(
            status="empty_pointcloud",
            mask_pixel_count=mask_pixel_count,
            valid_depth_pixel_count=valid_depth_pixel_count,
            raw_point_count=0,
            filtered_point_count=0,
            clustered_point_count=0,
        )

    if config.voxel_size_m > 0:
        pcd = pcd.voxel_down_sample(float(config.voxel_size_m))
    if len(pcd.points) >= max(32, config.outlier_nb_neighbors):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(config.outlier_nb_neighbors),
            std_ratio=float(config.outlier_std_ratio),
        )
    filtered_point_count = int(len(pcd.points))
    if filtered_point_count < int(config.min_point_count):
        return TargetGeometryEstimate(
            status="insufficient_points_after_filter",
            mask_pixel_count=mask_pixel_count,
            valid_depth_pixel_count=valid_depth_pixel_count,
            raw_point_count=raw_point_count,
            filtered_point_count=filtered_point_count,
            clustered_point_count=filtered_point_count,
        )

    pcd = _select_largest_cluster(pcd, config)
    clustered_point_count = int(len(pcd.points))
    if clustered_point_count < int(config.min_point_count):
        return TargetGeometryEstimate(
            status="insufficient_points_after_cluster",
            mask_pixel_count=mask_pixel_count,
            valid_depth_pixel_count=valid_depth_pixel_count,
            raw_point_count=raw_point_count,
            filtered_point_count=filtered_point_count,
            clustered_point_count=clustered_point_count,
        )

    obb = pcd.get_oriented_bounding_box(robust=True)
    rotation_matrix = np.asarray(obb.R, dtype=np.float32)
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1.0
    rotation = Rotation.from_matrix(rotation_matrix)
    return TargetGeometryEstimate(
        status="ok",
        mask_pixel_count=mask_pixel_count,
        valid_depth_pixel_count=valid_depth_pixel_count,
        raw_point_count=raw_point_count,
        filtered_point_count=filtered_point_count,
        clustered_point_count=clustered_point_count,
        geometry=TargetGeometry(
            center_xyz_m=np.asarray(obb.center, dtype=np.float32).astype(float).tolist(),
            rotation_matrix=rotation_matrix.astype(float).tolist(),
            euler_xyz_deg=rotation.as_euler("xyz", degrees=True).astype(float).tolist(),
            quaternion_xyzw=rotation.as_quat().astype(float).tolist(),
            extent_xyz_m=np.asarray(obb.extent, dtype=np.float32).astype(float).tolist(),
            point_count=int(len(pcd.points)),
            obb_corners_xyz_m=np.asarray(obb.get_box_points(), dtype=np.float32).astype(float).tolist(),
        ),
    )


def annotate_geometry(
    image_bgr: np.ndarray,
    estimate: TargetGeometryEstimate | None,
    intrinsics: CameraIntrinsics,
    axis_scale_m: float,
) -> np.ndarray:
    annotated = image_bgr.copy()
    if estimate is None:
        _draw_geometry_text(
            annotated,
            [
                "geometry: unavailable",
            ],
        )
        return annotated

    status_lines = [
        f"geometry: {estimate.status}",
        f"depth px: {estimate.valid_depth_pixel_count}/{estimate.mask_pixel_count}",
        f"points: {estimate.raw_point_count}->{estimate.filtered_point_count}->{estimate.clustered_point_count}",
    ]
    geometry = estimate.geometry
    if geometry is None:
        _draw_geometry_text(annotated, status_lines)
        return annotated

    K = intrinsics.matrix
    corners = np.asarray(geometry.obb_corners_xyz_m, dtype=np.float32)
    projected = _project_points(K, corners)
    if projected is not None:
        _draw_obb_edges(annotated, projected, color=(0, 255, 255), thickness=2)

    center = np.asarray(geometry.center_xyz_m, dtype=np.float32)
    R = np.asarray(geometry.rotation_matrix, dtype=np.float32)
    axes = np.stack(
        [
            center,
            center + R[:, 0] * float(axis_scale_m),
            center + R[:, 1] * float(axis_scale_m),
            center + R[:, 2] * float(axis_scale_m),
        ],
        axis=0,
    )
    projected_axes = _project_points(K, axes)
    if projected_axes is not None:
        origin = tuple(projected_axes[0].astype(int).tolist())
        for idx, color in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.line(
                annotated,
                origin,
                tuple(projected_axes[idx].astype(int).tolist()),
                color,
                3,
                cv2.LINE_AA,
            )

    tx, ty, tz = geometry.center_xyz_m
    rx, ry, rz = geometry.euler_xyz_deg
    text_lines = [
        *status_lines,
        f"xyz(m): {tx:+.4f} {ty:+.4f} {tz:+.4f}",
        f"rpy(deg): {rx:+.1f} {ry:+.1f} {rz:+.1f}",
        f"points: {geometry.point_count}",
    ]
    _draw_geometry_text(annotated, text_lines)
    return annotated


def _point_cloud_from_masked_depth(
    masked_depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth_m: float,
) -> o3d.geometry.PointCloud:
    depth_image = o3d.geometry.Image(masked_depth_m.astype(np.float32))
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(intrinsics.ppx),
        cy=float(intrinsics.ppy),
    )
    return o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        pinhole,
        np.eye(4, dtype=np.float64),
        1.0,
        float(max_depth_m),
        1,
        True,
    )


def _select_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    config: Open3DGeometryConfig,
) -> o3d.geometry.PointCloud:
    if len(pcd.points) < max(32, int(config.dbscan_min_points)):
        return pcd
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(config.dbscan_eps_m),
            min_points=int(config.dbscan_min_points),
            print_progress=False,
        )
    )
    valid = labels >= 0
    if not valid.any():
        return pcd
    largest_label = int(np.bincount(labels[valid]).argmax())
    indices = np.where(labels == largest_label)[0]
    return pcd.select_by_index(indices.tolist())


def _project_points(K: np.ndarray, points_xyz: np.ndarray) -> np.ndarray | None:
    if len(points_xyz) == 0:
        return None
    zs = points_xyz[:, 2:3]
    valid = zs[:, 0] > 1e-6
    if not valid.all():
        return None
    projected = (K @ points_xyz.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return projected.astype(np.float32)


def _draw_obb_edges(image_bgr: np.ndarray, projected_xy: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    edges = [
        (0, 1), (1, 7), (7, 2), (2, 0),
        (3, 6), (6, 4), (4, 5), (5, 3),
        (0, 3), (1, 6), (7, 4), (2, 5),
    ]
    for start, end in edges:
        p1 = tuple(projected_xy[start].astype(int).tolist())
        p2 = tuple(projected_xy[end].astype(int).tolist())
        cv2.line(image_bgr, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_geometry_text(image_bgr: np.ndarray, text_lines: list[str]) -> None:
    y = 58
    for line in text_lines:
        cv2.putText(
            image_bgr,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
