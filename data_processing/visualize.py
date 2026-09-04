import argparse
import os
import warnings
import numpy as np
import matplotlib
# Force headless backend for SLURM
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

import torch

from typing import Tuple, Dict
import os, glob, re, time
import pickle
# Own imports
try:
    from data_processing.sample import Sample
    from data_processing.utils import iter_results_db
except ImportError:
    from sample import Sample
    from utils import iter_results_db

# --- Visualization Constants ---
AXES_STR = ['X', 'Y', 'Z']

# Enhanced bright colors for dark background
BOX_COLORS = {
    'car': '#00ff88',        # Bright green
    'pedestrian': '#ff6b6b', # Coral red
    'cyclist': '#4ecdc4',    # Teal
    'bus': '#4ecdc4',    # Teal,
    'trailer': '#4ecdc4',    # Teal,
    'construction_vehicle': '#4ecdc4',    # Teal,
    'motorcycle': '#ff6b6b', # Coral red
    'bicycle': '#ff6b6b', # Coral red
    'traffic_cone': '#cf34eb', # Purple
    'barrier': '#cf34eb', # Purple
}

CLASS_NAMES = {
    "Kitti": ['pedestrian', 'cyclist', 'car'],
    "NuScenes": ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'],
    "Waymo": ['car', 'pedestrian', 'cyclist'],
}

DATASET_CLIP = {
    "Kitti": {"x": (0, 70.4),  "y": (-40, 40),  "z": (-3, 2)},
    "NuScenes": {"x": (-50, 50),  "y": (-50, 50),  "z": (-5, 4)},
    "Waymo": {"x": (-75, 75),  "y": (-75, 75),  "z": (-2, 5)},
}
BACKGROUND_ALPHA = 0.3

# Build a normalized mapping once (same as drawing)
KEYMAP = {k.lower(): k for k in BOX_COLORS.keys()}  # 'car' -> 'Car', etc.

def color_for(label_lower: str):
    key = KEYMAP.get(label_lower.lower())
    return BOX_COLORS[key] if key is not None else "black"


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Adversarial Point Clouds')
    parser.add_argument('--output', type=str, help='Directory to save images')
    parser.add_argument('--db-path', type=str, dest="db_path", help='Path to where the attack results are stored')
    parser.add_argument('--samples', nargs="+", default=None, help='Samples to be visualized')
    parser.add_argument('--adv', action='store_true', help='Compare original to adversarial pc?')
    parser.add_argument('--raw', action='store_true', help='Plot raw point cloud?')
    parser.add_argument('--objects', action='store_true', help='Should all objects be visualized with inner/outer points?')
    parser.add_argument('--reduced', action='store_true',help="Reduced Data?")
    parser.add_argument('--show-score-thr', type=float, default=0.1, help='Score threshold for visualizing predictions')
    parser.add_argument('--color-mode', type=str, default='depth', 
                        choices=['depth', 'height', 'intensity', 'density'],
                        help='Point coloring mode: depth, height, intensity, or density')
    parser.add_argument('--dark-theme', action='store_true', default=True,
                        help='Use dark background theme (better for adversarial visualization)')
    parser.add_argument('--light-theme', action='store_true',
                        help='Use light background theme')
    parser.add_argument('--points-keep-ratio', type=float, default=1.0,
                        help='Ratio of points to display (1.0 = all points, 0.5 = half)')
    parser.add_argument('--point-size', type=float, default=0.5,
                        help='Size of points in visualization (default: 0.5)')
    parser.add_argument('--no-3d', action='store_true',
                    help='Skip 3D visualization output')
    parser.add_argument('--no-bev', action='store_true',
                        help='Skip BEV (Bird\'s Eye View) output')
    parser.add_argument('--bev-dpi', type=int, default=300,
                        help='DPI for BEV output (default: 300, max recommended: 600)')
    parser.add_argument('--bev-figsize', type=float, nargs=2, default=[24, 20],
                        help='Figure size for BEV in inches (width height)')
    parser.add_argument('--input_suffix', default="", help='if the input file has a suffix that deviates fron standart', type=str)
    parser.add_argument('--no-legend', dest="no_legend", action='store_true',help="Show legend?")
    args = parser.parse_args()
    
    # Handle theme flags
    if args.light_theme:
        args.dark_theme = False
    
    return args

def main():
    args = parse_args()
    
    # Ensure output directory exists
    if args.output is None:
        output = "visualizations"
    else:
        output = args.output
    os.makedirs(output, exist_ok=True)
    base_path = args.db_path
    
    print(f"=" * 60)
    print(f"Enhanced Adversarial Point Cloud Visualizer")
    print(f"=" * 60)
    if args.samples:
        print(f"Visualizing specified Samples: {args.samples}")
        vis_samples = set(str(s) for s in args.samples)
    else:
        print(f"Visualizing first Sample.")
        vis_samples = None
    if args.objects:
        print(f"Visualizing all objects")
    print(f"Reading data from: {base_path}")
    print(f"Color mode: {args.color_mode}")
    print(f"Theme: {'Dark' if args.dark_theme else 'Light'}")
    print(f"Output directory: {output}")
    print(f"=" * 60)
    
    # Iterate through samples from db
    for res in iter_results_db(base_path):    
        sample_id = str(res["name"])
        dataset = res["dataset"]
        # Skip if not goal of visualization
        if vis_samples is not None and sample_id not in vis_samples:
            continue
        print("Visualizing Sample: ", sample_id)
        # compute sample data for later
        sample = Sample(res)
        # Extract Data
        gt_boxes_corners = sample.gt_boxes.corners
        gt_labels = sample.gt_labels

        if args.adv:
            points = sample.adv_points
            pred_instances = res["adv_result"][0]["pts_bbox"]
            scores = np.array([float(s.item()) if isinstance(s, torch.Tensor) else float(s) for s in res["adv_result"][0]["pts_bbox"]["scores_3d"]])
            mask = scores > args.show_score_thr
        else:
            points = sample.points
            pred_instances = res["result"][0]["pts_bbox"]
            scores = np.array([float(s.item()) if isinstance(s, torch.Tensor) else float(s) for s in res["result"][0]["pts_bbox"]["scores_3d"]])
            mask = scores > args.show_score_thr
            
        pred_bbox_corners = pred_instances["boxes_3d"][mask].corners
        pred_bbox_labels = pred_instances["labels_3d"][mask]
        
        # Convert tensors to numpy if needed
        if isinstance(gt_boxes_corners, torch.Tensor):
            gt_boxes_corners = gt_boxes_corners.detach().cpu().numpy()
        else:
            gt_boxes_corners = np.asarray(gt_boxes_corners)

        if isinstance(pred_bbox_corners, torch.Tensor):
            pred_bbox_corners = pred_bbox_corners.detach().cpu().numpy()
        else:
            pred_bbox_corners = np.asarray(pred_bbox_corners)

        if isinstance(pred_bbox_labels, torch.Tensor):
            pred_bbox_labels = pred_bbox_labels.detach().cpu().numpy()
        else:
            pred_bbox_labels = np.asarray(pred_bbox_labels)

        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        else:
            points = np.asarray(points)

        if args.adv:
            compare_adv(
                sample=sample,
                dataset=dataset,
                output_dir=output,
                show_thr = args.show_score_thr,
                filename_stem=sample_id,
                color_mode=args.color_mode,
                dark_theme=args.dark_theme,
                point_size=1,
                points_keep_ratio=args.points_keep_ratio,
                bev_dpi=args.bev_dpi,
                bev_figsize=tuple(args.bev_figsize),
                no_legend=args.no_legend
            )
        
        display_frame_statistics(
            point_cloud=points,
            gt_boxes_corners=gt_boxes_corners,
            pred_boxes_corners=pred_bbox_corners,
            gt_labels_numeric=gt_labels,
            pred_labels_numeric=pred_bbox_labels,
            dataset=dataset,
            output_dir=output,
            filename_stem=sample_id,
            color_mode=args.color_mode,
            dark_theme=args.dark_theme,
            point_size=args.point_size,
            points_keep_ratio=args.points_keep_ratio,
            output_3d=not args.no_3d,
            output_bev=not args.no_bev,
            output_obj=args.objects,
            bev_dpi=args.bev_dpi,
            bev_figsize=tuple(args.bev_figsize),
            sample = sample,
            no_legend=args.no_legend,
            raw = args.raw
        )

        if vis_samples is None:
            # Only visualize first sample if none specified
            break

    print(f"\n{'=' * 60}")
    print(f"Visualization complete!")
    print(f"Images saved to: {output}")
    print(f"{'=' * 60}")


def display_frame_statistics(point_cloud, gt_boxes_corners, pred_boxes_corners, 
                             gt_labels_numeric, pred_labels_numeric, dataset,
                             output_dir, filename_stem, 
                             color_mode='depth', dark_theme=True,
                             points_keep_ratio=1.0, point_size=0.25,
                             output_3d=True, output_bev=True, output_obj=True,
                             bev_dpi=300, bev_figsize=(24, 20),
                             sample=None, no_legend=False, raw=False):
    """
    Saves enhanced visualizations comparing ground truth and predicted boxes.
    
    Args:
        output_3d: Whether to output 3D visualization
        output_bev: Whether to output BEV (XY) projection
        bev_dpi: DPI for BEV output (higher = better quality)
        bev_figsize: Figure size for BEV output (width, height in inches)
    """
    # 1. Convert labels
    class_list = CLASS_NAMES.get(dataset, CLASS_NAMES.get("NuScenes"))
    LABEL_MAP = np.array(class_list)

    gt_ids = np.asarray([int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in gt_labels_numeric]).reshape(-1)
    pred_ids = np.asarray([int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in pred_labels_numeric]).reshape(-1)

    gt_labels_string = LABEL_MAP[gt_ids] if len(gt_ids) > 0 else np.array([])
    pred_labels_string = LABEL_MAP[pred_ids] if len(pred_ids) > 0 else np.array([])

    # 2. Subsample points
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    else:
        point_cloud = np.asarray(point_cloud)

    points_step = int(1. / points_keep_ratio) if points_keep_ratio > 0 else 1
    velo_range = range(0, point_cloud.shape[0], points_step)
    velo_frame = point_cloud[velo_range, :]

    # 3. Get colors based on mode
    print(f"  Computing {color_mode} colors...")
    colors, color_values, color_label = get_point_colors(velo_frame, mode=color_mode)
    
    cmap_names = {
        'depth': 'plasma',
        'height': 'viridis', 
        'intensity': 'hot',
        'density': 'coolwarm'
    }
    cmap_name = cmap_names.get(color_mode, 'plasma')

    # --- Create Legend ---
    legend_elements = make_dynamic_legend(
        dataset=dataset,
        CLASS_NAMES=CLASS_NAMES,
        box_colors=BOX_COLORS,
        gt_labels_string=gt_labels_string,
        pred_labels_string=pred_labels_string,
        show_only_present=True,
        include_pred=True,
    )

    dataset_clip = DATASET_CLIP.get(dataset, None)

    (xlim, ylim, zlim) = compute_dynamic_limits(
        velo_frame[:, :3],
        gt_boxes_corners=gt_boxes_corners,
        pred_boxes_corners=pred_boxes_corners,
        percentiles=(1, 99),
        margin=2.0,
        z_margin=1.0,
        dataset_clip=dataset_clip,
    )

    # --- Plot 1: 3D View (Optional) ---
    if output_3d:
        fig_3d = plt.figure(figsize=(16, 12))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.view_init(elev=30, azim=-45)
        
        if dark_theme:
            fig_3d.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig_3d, ax_3d, is_3d=True)
        
        ax_3d.scatter(
            velo_frame[:, 0], velo_frame[:, 1], velo_frame[:, 2],
            c=colors, s=point_size, alpha=0.8
        )
    
        ax_3d.set_title(f'3D LiDAR Scan: {filename_stem} | Color: {color_mode}', fontsize=12)
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_xlim3d(*xlim)
        ax_3d.set_ylim3d(*ylim)
        ax_3d.set_zlim3d(*zlim)
        
        for corners, label in zip(gt_boxes_corners, gt_labels_string):
            if label in BOX_COLORS:
                draw_box(ax_3d, corners.T, axes=[0, 1, 2],
                        color=BOX_COLORS[label], linestyle='-', linewidth=3.0)

        for corners, label in zip(pred_boxes_corners, pred_labels_string):
            if label in BOX_COLORS:
                draw_box(ax_3d, corners.T, axes=[0, 1, 2],
                        color=BOX_COLORS[label], linestyle='--', linewidth=2.5)

        add_colorbar(fig_3d, ax_3d, cmap_name, color_values, color_label, dark_theme)
        
        if not no_legend:
            legend = ax_3d.legend(handles=legend_elements, loc='upper left', fontsize=30)
            if dark_theme:
                legend.get_frame().set_facecolor('#2d2d44')
                legend.get_frame().set_edgecolor('white')
                for text in legend.get_texts():
                    text.set_color('white')
        
        output_path_3d = os.path.join(output_dir, f'{filename_stem}_3d_{color_mode}.png')
        fig_3d.tight_layout()
        fig_3d.savefig(output_path_3d, dpi=150, bbox_inches='tight')
        plt.close(fig_3d)
        print(f"  Saved: {output_path_3d}")

    # --- Plot 2: High-Resolution BEV (XY) Projection ---
    if output_bev:
        fig_bev = plt.figure(figsize=bev_figsize, dpi=bev_dpi)
        ax_bev = fig_bev.add_subplot(111)
        
        if dark_theme:
            fig_bev.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig_bev, ax_bev, is_3d=False)
        
        # Scatter points
        scatter = ax_bev.scatter(
            velo_frame[:, 0],
            velo_frame[:, 1],
            c=colors,
            s=point_size,
            alpha=0.8
        )
        
        ax_bev.set_title(f'Bird\'s Eye View: {filename_stem} | Color: {color_mode}', 
                         fontsize=18, pad=15)
        ax_bev.set_xlabel('X (m)', fontsize=14)
        ax_bev.set_ylabel('Y (m)', fontsize=14)
        ax_bev.set_xlim(*xlim)
        ax_bev.set_ylim(*ylim)
        ax_bev.set_aspect('equal')
        ax_bev.tick_params(labelsize=12)
        
        # Draw Ground Truth boxes (Solid, thick)
        for corners, label in zip(gt_boxes_corners, gt_labels_string):
            if label in BOX_COLORS:
                draw_box(ax_bev, corners.T, axes=[0, 1],
                        color=BOX_COLORS[label], linestyle='-', linewidth=3.0)

        # Draw Prediction boxes (Dashed)
        for corners, label in zip(pred_boxes_corners, pred_labels_string):
            if label in BOX_COLORS:
                draw_box(ax_bev, corners.T, axes=[0, 1],
                        color=BOX_COLORS[label], linestyle='--', linewidth=2.5)

        # Add colorbar
        add_colorbar(fig_bev, ax_bev, cmap_name, color_values, color_label, dark_theme)
        
        # Legend
        if not no_legend:
            legend = ax_bev.legend(handles=legend_elements, loc='upper right', fontsize=30)
            if dark_theme:
                legend.get_frame().set_facecolor('#2d2d44')
                legend.get_frame().set_edgecolor('white')
                for text in legend.get_texts():
                    text.set_color('white')
        
        output_path_bev = os.path.join(output_dir, f'{filename_stem}_bev_{color_mode}.png')
        fig_bev.tight_layout()
        fig_bev.savefig(output_path_bev, dpi=bev_dpi, bbox_inches='tight')
        plt.close(fig_bev)
        print(f"  Saved: {output_path_bev}")

    if raw:
        fig_bev = plt.figure(figsize=bev_figsize, dpi=bev_dpi)
        ax_bev = fig_bev.add_subplot(111)
        
        if dark_theme:
            fig_bev.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig_bev, ax_bev, is_3d=False)
        
        # Scatter points
        scatter = ax_bev.scatter(
            velo_frame[:, 0],
            velo_frame[:, 1],
            c=colors,
            s=point_size,
            alpha=0.8
        )
        
        ax_bev.set_title(f'Bird\'s Eye View: {filename_stem}', 
                         fontsize=18, pad=15)
        ax_bev.set_xlim(*xlim)
        ax_bev.set_ylim(*ylim)
        ax_bev.set_aspect('equal')
        ax_bev.tick_params(labelsize=12)

        output_path_raw = os.path.join(output_dir, f'{filename_stem}_raw.png')
        fig_bev.tight_layout()
        fig_bev.savefig(output_path_raw, dpi=bev_dpi, bbox_inches='tight')
        plt.close(fig_bev)
        print(f"  Saved: {output_path_raw}")

    if output_obj and sample is not None:
        visualize_object_inner_outer(
            sample.points, 
            sample.points_per_obj,
            sample.inner_points,
            sample.outer_points,
            sample.inner_boxes,
            sample.outer_boxes,
            output_dir,
            filename_stem)
        print(f"  Saved objects + inner/outer!")
        visualize_object_adv_comp(
            sample,
            output_dir,
            filename_stem
        )
        visualize_object(
            sample,
            True,
            output_dir,
            filename_stem
        )
        visualize_object(
            sample,
            False,
            output_dir,
            filename_stem
        )
        print(f"  Saved objects + adv comparison!!")

def visualize_object_adv_comp(
    sample,
    output_path, 
    filename_stem,
    dark_theme=True,
    point_size=2.0,
    view=(30, -45),
    margin=0.5,
    grid = False,
):
    """
    compares adversarial with clean object
    """
    outer_box = sample.outer_boxes
    if len(outer_box) == 0:
        return
    points_per_obj = sample.points_per_obj
    adv_points_per_obj = sample.adv_points_per_obj

    gt_box_corners = outer_box.corners.detach().cpu().numpy() if isinstance(outer_box.corners, torch.Tensor) else np.asarray(outer_box.corners)
    pts = sample.points.detach().cpu().numpy() if isinstance(sample.points, torch.Tensor) else np.asarray(sample.points)
    adv_pts = sample.adv_points.detach().cpu().numpy() if isinstance(sample.adv_points, torch.Tensor) else np.asarray(sample.adv_points)

    for i, (obj_p, adv_obj_p, gt_corner) in enumerate(zip(points_per_obj, adv_points_per_obj, gt_box_corners)):
        obj_p_np = obj_p.detach().cpu().numpy() if isinstance(obj_p, torch.Tensor) else np.asarray(obj_p)
        adv_obj_p_np = adv_obj_p.detach().cpu().numpy() if isinstance(adv_obj_p, torch.Tensor) else np.asarray(adv_obj_p)

        (xlim, ylim, zlim) = compute_dynamic_limits(
            obj_p_np[:, :3],
            gt_boxes_corners=gt_corner,
            pred_boxes_corners=gt_corner,
            percentiles=(5, 95),
            margin=0.0,
            z_margin=0.0,
            zoom = 0.7,
        )
        diff_points_i = sample.diff_pc_list(obj_p_np, adv_obj_p_np)
        obj_points_i = obj_p_np

        gt_box_corners_i = np.asarray(gt_corner)
        if gt_box_corners_i.shape == (3, 8):
            gt_box_for_draw = gt_box_corners_i
        else:
            gt_box_for_draw = gt_box_corners_i.T

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        if dark_theme:
            fig.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig, ax, is_3d=True)

        draw_box(ax, gt_box_for_draw, axes=[0, 1, 2], color="#2ecc71", linestyle="-", linewidth=1.0)

        outer_corners_8x3 = gt_box_for_draw.T
        mins = outer_corners_8x3.min(axis=0) - margin
        maxs = outer_corners_8x3.max(axis=0) + margin

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        set_axes_equal_3d(ax)

        target = outer_corners_8x3.mean(axis=0)
        elev, azim = view_from_lidar(target, lidar_xyz=(0.0, 0.0, 0.0))
        ax.view_init(elev=elev, azim=azim)

        box_diag = np.linalg.norm(maxs - mins)
        diff_push = 0.01 * box_diag

        bg_pts = remove_points_inside_obb(pts[:, :3], outer_corners_8x3)
        bg_adv_pts = remove_points_inside_obb(adv_pts[:, :3], outer_corners_8x3)
        diff_bg_pts = sample.diff_pc_list(bg_pts, bg_adv_pts)

        if len(diff_bg_pts) > 0:
            diff_bg_pts = nudge_toward_camera(diff_bg_pts[:, :3], elev, azim, amount=diff_push)

        ax.scatter(
            bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2],
            c="#b0b0b0" if not dark_theme else "#fcfafa",
            s=point_size,
            alpha=BACKGROUND_ALPHA,
            linewidths=0, 
            depthshade=False
        )
        if len(diff_bg_pts) > 0:
            ax.scatter(
                diff_bg_pts[:, 0], diff_bg_pts[:, 1], diff_bg_pts[:, 2],
                c="#e67e22",
                s=point_size*1.5,
                alpha=0.65,
                linewidths=0, 
                depthshade=False
            )

        ax.scatter(
            obj_points_i[:, 0], obj_points_i[:, 1], obj_points_i[:, 2],
            c="#2ecc71",
            s=point_size * 1.5,
            alpha=0.75,
            linewidths=0
        )

        if len(diff_points_i) > 0:
            diff_points_i_nudged = nudge_toward_camera(diff_points_i[:, :3], elev, azim, amount=diff_push)
            ax.scatter(
                diff_points_i_nudged[:, 0], diff_points_i_nudged[:, 1], diff_points_i_nudged[:, 2],
                c="#e74c3c",
                s=point_size * 2.5,
                alpha=0.95,
                linewidths=0,
                depthshade=False
            )

        ax.set_title(f"Sample: {filename_stem} | Object: {i}", fontsize=12, pad=20, color="white" if dark_theme else "black")
        out_file = os.path.join(output_path, f"{filename_stem}_obj_{i}_adv_comp.png")
        fig.tight_layout()
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)


def visualize_object(
    sample,
    adv,
    output_path, 
    filename_stem,
    dark_theme=True,
    point_size=2.0,
    view=(30, -45),
    margin=0.5,
    grid = False,
):
    """
    compares adversarial with clean object
    """
    outer_box = sample.outer_boxes
    if len(outer_box) == 0:
        return
    if adv:
        points_per_obj = sample.adv_points_per_obj
        pts = sample.adv_points.detach().cpu().numpy() if isinstance(sample.adv_points, torch.Tensor) else np.asarray(sample.adv_points)
        is_adv = "adv"
    else:
        points_per_obj = sample.points_per_obj
        pts = sample.points.detach().cpu().numpy() if isinstance(sample.points, torch.Tensor) else np.asarray(sample.points)
        is_adv = "clean"

    gt_box_corners = outer_box.corners.detach().cpu().numpy() if isinstance(outer_box.corners, torch.Tensor) else np.asarray(outer_box.corners)

    for i, (obj_p, gt_corner) in enumerate(zip(points_per_obj, gt_box_corners)):
        obj_p_np = obj_p.detach().cpu().numpy() if isinstance(obj_p, torch.Tensor) else np.asarray(obj_p)
        (xlim, ylim, zlim) = compute_dynamic_limits(
            obj_p_np[:, :3],
            gt_boxes_corners=gt_corner,
            pred_boxes_corners=gt_corner,
            percentiles=(5, 95),
            margin=0.0,
            z_margin=0.0,
            zoom = 0.7,
        )
        obj_points_i = obj_p_np

        gt_box_corners_i = np.asarray(gt_corner)
        if gt_box_corners_i.shape == (3, 8):
            gt_box_for_draw = gt_box_corners_i
        else:
            gt_box_for_draw = gt_box_corners_i.T

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        if dark_theme:
            fig.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig, ax, is_3d=True)

        draw_box(ax, gt_box_for_draw, axes=[0, 1, 2], color="#2ecc71", linestyle="-", linewidth=1.0)

        outer_corners_8x3 = gt_box_for_draw.T
        mins = outer_corners_8x3.min(axis=0) - margin
        maxs = outer_corners_8x3.max(axis=0) + margin

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        set_axes_equal_3d(ax)

        target = outer_corners_8x3.mean(axis=0)
        elev, azim = view_from_lidar(target, lidar_xyz=(0.0, 0.0, 0.0))
        ax.view_init(elev=elev, azim=azim)

        bg_pts = remove_points_inside_obb(pts[:, :3], outer_corners_8x3)

        ax.scatter(
            bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2],
            c="#b0b0b0" if not dark_theme else "#fcfafa",
            s=point_size,
            alpha=BACKGROUND_ALPHA,
            linewidths=0, 
            depthshade=False
        )

        ax.scatter(
            obj_points_i[:, 0], obj_points_i[:, 1], obj_points_i[:, 2],
            c="#2ecc71",
            s=point_size * 1.5,
            alpha=0.75,
            linewidths=0
        )

        ax.set_title(f"Sample: {filename_stem} | Object: {i} | Mode: {is_adv}", fontsize=12, pad=20, color="white" if dark_theme else "black")
        out_file = os.path.join(output_path, f"{filename_stem}_obj_{i}_{is_adv}.png")
        fig.tight_layout()
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)


def nudge_toward_camera(points_xyz, elev_deg, azim_deg, amount=0.02):
    """
    Shifts points slightly toward the camera along the line of sight.
    amount: distance in meters (e.g. 0.02 = 2 cm)
    """
    if len(points_xyz) == 0:
        return points_xyz
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    # Camera position in standard spherical coords (direction from origin to camera)
    v_cam = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev)
    ], dtype=np.float32)
    v_cam = v_cam / np.linalg.norm(v_cam)
    return points_xyz + amount * v_cam


def remove_points_inside_obb(points, box_corners, eps=1e-6):
    """
    points: (N, 3)
    box_corners: (8, 3)
    Returns: points OUTSIDE the box, (M, 3)
    """
    if len(points) == 0:
        return points
    center = box_corners.mean(axis=0)
    u_x = box_corners[1] - box_corners[0]
    u_y = box_corners[3] - box_corners[0]
    u_z = box_corners[4] - box_corners[0]

    lx = np.linalg.norm(u_x)
    ly = np.linalg.norm(u_y)
    lz = np.linalg.norm(u_z)

    if lx < eps or ly < eps or lz < eps:
        return points

    ex = u_x / lx
    ey = u_y / ly
    ez = u_z / lz

    R = np.vstack([ex, ey, ez])

    p_rel = (points - center) @ R.T

    inside_mask = (
        (np.abs(p_rel[:, 0]) <= lx / 2.0) &
        (np.abs(p_rel[:, 1]) <= ly / 2.0) &
        (np.abs(p_rel[:, 2]) <= lz / 2.0)
    )

    return points[~inside_mask]


def visualize_object_inner_outer(
    points, obj_points, inner_points, outer_points,
    inner_box, outer_box,
    output_path, filename_stem,
    dark_theme=True,
    point_size=2.0,
    view=(30, -45),
    margin=-0.1,
    grid = False,
):
    """
    obj_points, inner_points, outer_points: (N,3) arrays for ONE object
    inner_box_corners, outer_box_corners: corners shaped (8,3) OR (3,8) depending on your draw_box usage
    """
    if len(inner_box) == 0 or len(outer_box) == 0:
        return
    inner_box_corners = inner_box.corners.detach().cpu().numpy() if isinstance(inner_box.corners, torch.Tensor) else np.asarray(inner_box.corners)
    outer_box_corners = outer_box.corners.detach().cpu().numpy() if isinstance(outer_box.corners, torch.Tensor) else np.asarray(outer_box.corners)
    pts = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else np.asarray(points)

    for i, (obj_p, inn_pts, out_pts, inn_corner, out_corner) in enumerate(zip(obj_points, inner_points, outer_points, inner_box_corners, outer_box_corners)):
        obj_p_np = obj_p.detach().cpu().numpy() if isinstance(obj_p, torch.Tensor) else np.asarray(obj_p)
        inn_pts_np = inn_pts.detach().cpu().numpy() if isinstance(inn_pts, torch.Tensor) else np.asarray(inn_pts)
        out_pts_np = out_pts.detach().cpu().numpy() if isinstance(out_pts, torch.Tensor) else np.asarray(out_pts)

        (xlim, ylim, zlim) = compute_dynamic_limits(
            obj_p_np[:, :3],
            gt_boxes_corners=out_corner,
            pred_boxes_corners=inn_corner,
            percentiles=(0, 100),
            margin=margin,
            z_margin=margin,
        )

        obj_points_i = obj_p_np
        inner_points_i = inn_pts_np
        outer_points_i = out_pts_np

        inner_box_corners_i = np.asarray(inn_corner)
        outer_box_corners_i = np.asarray(out_corner)

        if inner_box_corners_i.shape == (3, 8):
            inner_box_for_draw = inner_box_corners_i
        else:
            inner_box_for_draw = inner_box_corners_i.T

        if outer_box_corners_i.shape == (3, 8):
            outer_box_for_draw = outer_box_corners_i
        else:
            outer_box_for_draw = outer_box_corners_i.T

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        if dark_theme:
            fig.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig, ax, is_3d=True)

        outer_corners_8x3 = outer_box_for_draw.T
        mins = outer_corners_8x3.min(axis=0) - margin
        maxs = outer_corners_8x3.max(axis=0) + margin

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        set_axes_equal_3d(ax)

        target = outer_corners_8x3.mean(axis=0)
        elev, azim = view_from_lidar(target, lidar_xyz=(0.0, 0.0, 0.0))
        ax.view_init(elev=elev, azim=azim)

        bg_pts = remove_points_inside_obb(pts[:, :3], outer_corners_8x3)

        draw_box(ax, inner_box_for_draw, axes=[0, 1, 2], color="#2ecc71", linestyle="-", linewidth=1.0)
        draw_box(ax, outer_box_for_draw, axes=[0, 1, 2], color="#ff6b6b", linestyle="--", linewidth=1.0)

        ax.scatter(
            bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2],
            c="#b0b0b0" if not dark_theme else "#fcfafa",
            s=point_size,
            alpha=BACKGROUND_ALPHA,
            linewidths=0, 
            depthshade=False
        )

        ax.scatter(
            obj_points_i[:, 0], obj_points_i[:, 1], obj_points_i[:, 2],
            c="#2ecc71",
            s=point_size * 1.5,
            alpha=0.75,
            linewidths=0
        )

        if len(inner_points_i) > 0:
            ax.scatter(
                inner_points_i[:, 0], inner_points_i[:, 1], inner_points_i[:, 2],
                c="#3498db",
                s=point_size * 1.5,
                alpha=0.85,
                linewidths=0
            )

        if len(outer_points_i) > 0:
            ax.scatter(
                outer_points_i[:, 0], outer_points_i[:, 1], outer_points_i[:, 2],
                c="#e67e22",
                s=point_size * 1.5,
                alpha=0.85,
                linewidths=0
            )

        ax.set_title(f"Sample: {filename_stem} | Object: {i}", fontsize=12, pad=20, color="white" if dark_theme else "black")
        out_file = os.path.join(output_path, f"{filename_stem}_obj_{i}_inner_outer.png")
        fig.tight_layout()
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)


def set_axes_equal_3d(ax):
    """Make axes of 3D plot have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def view_from_lidar(target_xyz, lidar_xyz=(0.0, 0.0, 0.0)):
    """
    Computes camera view angles looking from lidar towards target object.
    """
    target = np.asarray(target_xyz)
    lidar = np.asarray(lidar_xyz)
    d = target - lidar

    dist_xy = np.hypot(d[0], d[1])
    elev = np.rad2deg(np.arctan2(d[2], max(dist_xy, 1e-6)))
    azim = np.rad2deg(np.arctan2(d[1], d[0]))
    return elev, azim


def compare_adv(sample, dataset, show_thr, output_dir, filename_stem, 
                color_mode='depth', dark_theme=True,
                points_keep_ratio=1.0, point_size=0.5,
                bev_dpi=300, bev_figsize=(24, 20),
                no_legend=False):
    """
    Compares the clean and adversarial point cloud and pred boxes
    """
    # Extract data
    point_cloud = sample.points
    adv_point_cloud = sample.adv_points
    adv_diff_points = sample.diff_pc_list(point_cloud, adv_point_cloud)
    gt_boxes_corners = sample.gt_boxes.corners
    gt_labels = sample.gt_labels

    pred_instances = sample.result["pts_bbox"]
    scores = np.array([float(s.item()) if isinstance(s, torch.Tensor) else float(s) for s in sample.result["pts_bbox"]["scores_3d"]])
    mask = scores > show_thr
    pred_boxes_corners = pred_instances["boxes_3d"][mask].corners
    pred_labels = pred_instances["labels_3d"][mask]

    adv_pred_instances = sample.adv_result["pts_bbox"]
    adv_scores = np.array([float(s.item()) if isinstance(s, torch.Tensor) else float(s) for s in sample.adv_result["pts_bbox"]["scores_3d"]])
    mask_adv = adv_scores > show_thr
    pred_boxes_corners_adv = adv_pred_instances["boxes_3d"][mask_adv].corners
    pred_labels_adv = adv_pred_instances["labels_3d"][mask_adv]

    # Convert tensors to numpy if needed
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    if isinstance(adv_diff_points, torch.Tensor):
        adv_diff_points = adv_diff_points.detach().cpu().numpy()
    if isinstance(gt_boxes_corners, torch.Tensor):
        gt_boxes_corners = gt_boxes_corners.detach().cpu().numpy()
    if isinstance(pred_boxes_corners, torch.Tensor):
        pred_boxes_corners = pred_boxes_corners.detach().cpu().numpy()
    if isinstance(pred_boxes_corners_adv, torch.Tensor):
        pred_boxes_corners_adv = pred_boxes_corners_adv.detach().cpu().numpy()

    # Convert labels
    class_list = CLASS_NAMES.get(dataset, CLASS_NAMES.get("NuScenes"))
    LABEL_MAP = np.array(class_list)

    gt_ids = np.asarray([int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in gt_labels]).reshape(-1)
    pred_ids = np.asarray([int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in pred_labels]).reshape(-1)
    pred_ids_adv = np.asarray([int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in pred_labels_adv]).reshape(-1)

    gt_labels_string = LABEL_MAP[gt_ids] if len(gt_ids) > 0 else np.array([])
    pred_labels_string = LABEL_MAP[pred_ids] if len(pred_ids) > 0 else np.array([])
    pred_labels_string_adv = LABEL_MAP[pred_ids_adv] if len(pred_ids_adv) > 0 else np.array([])

    fig_bev = plt.figure(figsize=bev_figsize, dpi=bev_dpi)
    ax_bev = fig_bev.add_subplot(111)

    if dark_theme:
        fig_bev.patch.set_facecolor('#1a1a2e')
        setup_dark_theme(fig_bev, ax_bev, is_3d=False)

    # 2. Subsample points
    points_step = int(1. / points_keep_ratio) if points_keep_ratio > 0 else 1
    velo_range = range(0, point_cloud.shape[0], points_step)
    velo_frame_clean = point_cloud[velo_range, :]

    if len(adv_diff_points) > 0:
        adv_velo_range = range(0, adv_diff_points.shape[0], points_step)
        velo_frame_adv = adv_diff_points[adv_velo_range, :]
    else:
        velo_frame_adv = np.empty((0, point_cloud.shape[1]))

    # 3. Get colors based on mode
    print(f"  Computing {color_mode} colors...")
    colors, color_values, color_label = get_point_colors(velo_frame_clean, mode=color_mode)
    
    cmap_names = {
        'depth': 'plasma',
        'height': 'viridis', 
        'intensity': 'hot',
        'density': 'coolwarm'
    }
    cmap_name = cmap_names.get(color_mode, 'plasma')

    # --- Create Legend ---
    legend_elements = make_dynamic_legend(
        dataset=dataset,
        CLASS_NAMES=CLASS_NAMES,
        box_colors=BOX_COLORS,
        gt_labels_string=gt_labels_string,
        pred_labels_string=pred_labels_string,
        show_only_present=True,
        include_pred=True,
    )

    dataset_clip = DATASET_CLIP.get(dataset, None)

    (xlim, ylim, zlim) = compute_dynamic_limits(
        velo_frame_clean[:, :3],
        gt_boxes_corners=gt_boxes_corners,
        pred_boxes_corners=pred_boxes_corners,
        percentiles=(1, 99),
        margin=2.0,
        z_margin=1.0,
        dataset_clip=dataset_clip,
    )

    # 1) Scatter Clean points
    sc_clean = ax_bev.scatter(
        velo_frame_clean[:, 0],
        velo_frame_clean[:, 1],
        c="#fcfafa",            # White
        s=point_size,
        alpha=1,
        linewidths=0,
        label="Points (clean)"
    )

    # Adv points
    if len(velo_frame_adv) > 0:
        adv_point_color = "#ff0000" if not dark_theme else "#e67e22"
        sc_adv = ax_bev.scatter(
            velo_frame_adv[:, 0],
            velo_frame_adv[:, 1],
            c=adv_point_color,
            s=point_size * 4,
            alpha=1,
            linewidths=0,
            label="Points (adv)"
        )

    ax_bev.set_title(
        f"BEV Overlay: {filename_stem} | Adversarial comparison",
        fontsize=18, pad=15
    )
    ax_bev.set_xlabel("X (m)", fontsize=14)
    ax_bev.set_ylabel("Y (m)", fontsize=14)
    ax_bev.set_xlim(*xlim)
    ax_bev.set_ylim(*ylim)
    ax_bev.set_aspect("equal")
    ax_bev.tick_params(labelsize=12)

    # 2) Draw boxes: GT, clean pred, adv pred
    for corners, label in zip(gt_boxes_corners, gt_labels_string):
        if label in BOX_COLORS:
            draw_box(
                ax_bev, corners.T, axes=[0, 1],
                color=BOX_COLORS[label],
                linestyle='-',
                linewidth=3.0
            )

    for corners, label in zip(pred_boxes_corners, pred_labels_string):
        if label in BOX_COLORS:
            draw_box(
                ax_bev, corners.T, axes=[0, 1],
                color=BOX_COLORS[label],
                linestyle='--',
                linewidth=2.5
            )

    for corners, label in zip(pred_boxes_corners_adv, pred_labels_string_adv):
        if label in BOX_COLORS:
            draw_box(
                ax_bev, corners.T, axes=[0, 1],
                color=BOX_COLORS[label],
                linestyle='-.',
                linewidth=2.5
            )

    # Legend
    if not no_legend:
        legend = ax_bev.legend(handles=legend_elements, loc='upper right', fontsize=30)
        if dark_theme:
            legend.get_frame().set_facecolor('#2d2d44')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')

    output_path_bev = os.path.join(output_dir, f'{filename_stem}_bev_comparison.png')
    fig_bev.tight_layout()
    fig_bev.savefig(output_path_bev, dpi=bev_dpi, bbox_inches='tight')
    plt.close(fig_bev)
    print(f"  Saved: {output_path_bev}")


def make_dynamic_legend(
    dataset,
    CLASS_NAMES,
    box_colors,
    *,
    gt_labels_string=None,
    pred_labels_string=None,
    show_only_present=False,
    include_pred=True,
    gt_style=("-", 2.8),
    pred_style=("--", 2.2),
):
    """
    Build legend handles dynamically from dataset class names and the color mapping actually used.
    """
    dataset_classes = [str(x) for x in CLASS_NAMES.get(dataset, CLASS_NAMES.get("NuScenes", []))]

    keymap = {str(k).strip().lower(): k for k in box_colors.keys()}

    present = None
    if show_only_present:
        present = set()
        if gt_labels_string is not None:
            present |= {str(x).strip().lower() for x in np.asarray(gt_labels_string).reshape(-1)}
        if pred_labels_string is not None:
            present |= {str(x).strip().lower() for x in np.asarray(pred_labels_string).reshape(-1)}

    gt_ls, gt_lw = gt_style
    pr_ls, pr_lw = pred_style

    legend_elements = []
    for cname in dataset_classes:
        cname_l = cname.strip().lower()

        if cname_l not in keymap:
            continue

        if present is not None and cname_l not in present:
            continue

        canonical_key = keymap[cname_l]
        color = box_colors[canonical_key]

        legend_elements.append(
            Line2D([0], [0], color=color, lw=gt_lw, linestyle=gt_ls, label=f"{cname} (GT)")
        )

        if include_pred:
            legend_elements.append(
                Line2D([0], [0], color=color, lw=pr_lw, linestyle=pr_ls, label=f"{cname} (Pred)")
            )

    return legend_elements


def compute_dynamic_limits(
    points_xyz,
    gt_boxes_corners=None,
    pred_boxes_corners=None,
    *,
    percentiles=(1.0, 99.0),
    margin=2.0,
    z_margin=1.0,
    dataset_clip=None,
    zoom = 1.0,
):
    """
    Compute dynamic axis limits from points (+ optional boxes), robustly.
    """
    if isinstance(points_xyz, torch.Tensor):
        pts = points_xyz.detach().cpu().numpy()
    else:
        pts = np.asarray(points_xyz)

    xs = [pts[:, 0]]
    ys = [pts[:, 1]]
    zs = [pts[:, 2]]

    def add_boxes(boxes):
        if boxes is None:
            return
        if isinstance(boxes, torch.Tensor):
            b = boxes.detach().cpu().numpy()
        else:
            b = np.asarray(boxes)
        if b.size == 0:
            return
        b = b.reshape(-1, 3)
        xs.append(b[:, 0])
        ys.append(b[:, 1])
        zs.append(b[:, 2])

    add_boxes(gt_boxes_corners)
    add_boxes(pred_boxes_corners)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    
    lo, hi = percentiles
    x0, x1 = np.percentile(x, [lo, hi])
    y0, y1 = np.percentile(y, [lo, hi])
    z0, z1 = np.percentile(z, [lo, hi])

    # Add margins
    x0 -= margin; x1 += margin
    y0 -= margin; y1 += margin
    z0 -= z_margin; z1 += z_margin

    # Optional dataset clip
    if dataset_clip is not None:
        if "x" in dataset_clip:
            x0, x1 = max(x0, dataset_clip["x"][0]), min(x1, dataset_clip["x"][1])
        if "y" in dataset_clip:
            y0, y1 = max(y0, dataset_clip["y"][0]), min(y1, dataset_clip["y"][1])
        if "z" in dataset_clip:
            z0, z1 = max(z0, dataset_clip["z"][0]), min(z1, dataset_clip["z"][1])

    # Avoid degenerate ranges
    if x1 - x0 < 1e-3: x0 -= 1; x1 += 1
    if y1 - y0 < 1e-3: y0 -= 1; y1 += 1
    if z1 - z0 < 1e-3: z0 -= 1; z1 += 1

    return (x0 * zoom, x1 * zoom), (y0 * zoom, y1 * zoom), (z0 * zoom, z1 * zoom)


# --- Point Cloud Coloring Functions ---

def get_point_colors_by_depth(points, cmap='plasma'):
    """Color points by distance from sensor origin."""
    distances = np.linalg.norm(points[:, :3], axis=1)
    norm = Normalize(vmin=distances.min(), vmax=distances.max())
    colormap = plt.get_cmap(cmap)
    return colormap(norm(distances)), distances, 'Distance (m)'


def get_point_colors_by_height(points, cmap='viridis'):
    """Color points by Z-height - good for seeing ground vs objects."""
    z_vals = points[:, 2]
    norm = Normalize(vmin=z_vals.min(), vmax=z_vals.max())
    colormap = plt.get_cmap(cmap)
    return colormap(norm(z_vals)), z_vals, 'Height (m)'


def get_point_colors_by_intensity(points, cmap='hot'):
    """Color points by intensity (4th channel if available)."""
    if points.shape[1] >= 4:
        intensity = points[:, 3]
    else:
        intensity = np.linalg.norm(points[:, :3], axis=1)
    norm = Normalize(vmin=intensity.min(), vmax=intensity.max())
    colormap = plt.get_cmap(cmap)
    return colormap(norm(intensity)), intensity, 'Intensity'


def get_point_colors_by_density(points, radius=0.5, cmap='coolwarm'):
    """Color by local point density - highlights adversarial clusters."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, :3])
    density = np.array([len(tree.query_ball_point(p, radius)) for p in points[:, :3]])
    norm = Normalize(vmin=density.min(), vmax=density.max())
    colormap = plt.get_cmap(cmap)
    return colormap(norm(density)), density, 'Local Density'


def get_point_colors(points, mode='depth'):
    """Get colors based on selected mode."""
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    else:
        points = np.asarray(points)

    color_funcs = {
        'depth': lambda p: get_point_colors_by_depth(p, 'plasma'),
        'height': lambda p: get_point_colors_by_height(p, 'viridis'),
        'intensity': lambda p: get_point_colors_by_intensity(p, 'hot'),
        'density': lambda p: get_point_colors_by_density(p, 0.5, 'coolwarm'),
    }
    return color_funcs.get(mode, color_funcs['depth'])(points)


# --- Drawing Functions ---

def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black', linestyle='-', linewidth=1.5):
    """
    Draws a bounding 3D box in a pyplot axis.
    vertices: (3, 8) or (8, 3) numpy array
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    else:
        vertices = np.asarray(vertices)
    if vertices.shape[0] != len(axes) and vertices.shape[1] == len(axes):
        vertices = vertices.T
    vertices = vertices[axes, :] 
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connections
    ] 
    
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=linewidth, linestyle=linestyle)


def setup_dark_theme(fig, ax, is_3d=True):
    """Apply dark theme styling to figure and axes."""
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Style axes
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.title.set_color('white')
    
    if is_3d:
        ax.zaxis.label.set_color('white')
        ax.tick_params(axis='z', colors='white')
        # Make panes transparent/dark
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.grid(True, alpha=0.3)
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        ax.grid(True, alpha=0.3, color='gray')


def add_colorbar(fig, ax, cmap, values, label, dark_theme=True):
    """Add a colorbar to the figure."""
    colormap = plt.get_cmap(cmap)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=values.min(), vmax=values.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(label, fontsize=10)
    
    if dark_theme:
        cbar.set_label(label, color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')


if __name__ == '__main__':
    main()
