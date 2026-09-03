import argparse
import os
import numpy as np
import matplotlib
# Force headless backend for SLURM
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

import torch

from typing import Tuple, Dict
import os, glob, re, time
import pickle
# Own imports
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
    parser.add_argument('--samples', nargs="+", required=True, help='Samples to be visualized')
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
    else:
        print(f"Visualizing first Sample.")
    if args.objects:
        print(f"Visualizing all objects")
    print(f"Reading data from: {base_path}")
    print(f"Color mode: {args.color_mode}")
    print(f"Theme: {'Dark' if args.dark_theme else 'Light'}")
    print(f"Output directory: {output}")
    print(f"=" * 60)
    
    vis_samples = set(args.samples)
    print("Samples that will be visualized: ", vis_samples)
    # Iterate through samples from pickle file
    for res in iter_results_db(base_path):    
        sample_id = str(res["name"])
        dataset = res["dataset"]
        # Skip if not goal of visualization
        if sample_id not in vis_samples:
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
            scores = np.array([s.item() for s in res["adv_result"][0]["pts_bbox"]["scores_3d"]])
            mask = scores > args.show_score_thr
        else:
            points = sample.points
            pred_instances = res["result"][0]["pts_bbox"]
            scores = np.array([s.item() for s in res["result"][0]["pts_bbox"]["scores_3d"]])
            mask = scores > args.show_score_thr
            
        pred_bbox_corners = pred_instances["boxes_3d"][mask].corners
        pred_bbox_labels = pred_instances["labels_3d"][mask]
        
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
    # --- build mapped strings robustly ---
    LABEL_MAP = np.array(CLASS_NAMES[dataset])

    gt_ids = np.asarray(gt_labels_numeric).astype(int).reshape(-1)
    pred_ids = np.asarray(pred_labels_numeric).astype(int).reshape(-1)

    gt_labels_string = LABEL_MAP[gt_ids]
    pred_labels_string = LABEL_MAP[pred_ids]

    # 2. Subsample points
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
        show_only_present=True,   # shows only classes that appear in this sample
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
        plt.savefig(output_path_3d, dpi=150, facecolor=fig_3d.get_facecolor(), bbox_inches='tight')
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
        plt.savefig(output_path_bev, dpi=bev_dpi, facecolor=fig_bev.get_facecolor(), bbox_inches='tight')
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

        # Add colorbar
        #add_colorbar(fig_bev, ax_bev, cmap_name, color_values, color_label, dark_theme)
        
        output_path_raw = os.path.join(output_dir, f'{filename_stem}_raw.png')
        fig_bev.tight_layout()
        plt.savefig(output_path_raw, dpi=bev_dpi, facecolor=fig_bev.get_facecolor(), bbox_inches='tight')
        plt.close(fig_bev)
        print(f"  Saved: {output_path_raw}")

    if output_obj:
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
    compares advesarial with clean object
    """
    # gt box
    outer_box = sample.outer_boxes
    # clean/adv points per obj
    points_per_obj = sample.points_per_obj
    adv_points_per_obj = sample.adv_points_per_obj

    gt_box_corners = outer_box.corners.cpu()
    pts = np.asarray(sample.points.cpu())
    adv_pts = np.asarray(sample.adv_points.cpu())
    for i,(obj_p, adv_obj_p, gt_corner) in enumerate(zip(points_per_obj, adv_points_per_obj, gt_box_corners)):
        (xlim, ylim, zlim) = compute_dynamic_limits(
            obj_p[:, :3],
            gt_boxes_corners=gt_corner,
            pred_boxes_corners=gt_corner,
            percentiles=(5, 95),
            margin=-0.0,
            z_margin=-0.0,
            zoom = 0.7,
        )
        # diff points
        diff_points_i = sample.diff_pc_list(obj_p, adv_obj_p)
        # print("amt. diff points: ", len(diff_points_i))
        # --- Normalize shapes ---
        obj_points_i = np.asarray(obj_p.cpu())
        # print("before amt. obj points: ", len(obj_points_i))
        # obj_points_i = sample.diff_pc_list(adv_obj_p, obj_p)
        # print("amt. obj points: ", len(obj_points_i))

        gt_box_corners = np.asarray(gt_corner)

        if gt_box_corners.shape == (3, 8):
            gt_box_for_draw = gt_box_corners
        else:
            gt_box_for_draw = gt_box_corners.T

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        if dark_theme:
            fig.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig, ax, is_3d=True)  # your existing helper


        # --- Boxes ---
        draw_box(ax, gt_box_for_draw, axes=[0, 1, 2], color="#2ecc71", linestyle="-", linewidth=1.0)

        # --- Limits: frame around OUTER box (plus margin) ---
        outer_corners_8x3 = gt_box_for_draw.T  # (8,3)
        mins = outer_corners_8x3.min(axis=0) - margin
        maxs = outer_corners_8x3.max(axis=0) + margin

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        set_axes_equal_3d(ax)

        # adjust view dynamically
        target = outer_corners_8x3.mean(axis=0)
        elev, azim = view_from_lidar(target, lidar_xyz=(0.0, 0.0, 0.0))
        ax.view_init(elev=elev, azim=azim)

        # choose an amount relative to box size (robust) to make sure that diff points are in foreground
        box_diag = np.linalg.norm(maxs - mins)
        diff_push = 0.01 * box_diag  # 1% of diagonal (tune 0.005–0.02)

        # --- Points: background object points ---
        # Remove background points inside GT box
        
        bg_pts = remove_points_inside_obb(pts[:, :3], outer_corners_8x3)
        bg_adv_pts = remove_points_inside_obb(adv_pts[:, :3], outer_corners_8x3)
        diff_bg_pts = sample.diff_pc_list(bg_pts, bg_adv_pts)

        diff_bg_pts = nudge_toward_camera(diff_bg_pts[:, :3], elev, azim, amount=diff_push)
        # print("before amt. points: ", len(pts))
        # print("amt. points: ", len(bg_pts))

        ax.scatter(
            bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2],
            c="#b0b0b0" if not dark_theme else "#fcfafa",
            s=point_size,
            alpha=BACKGROUND_ALPHA,
            linewidths=0, 
            depthshade=False
        )
        # adv background points
        ax.scatter(
            diff_bg_pts[:, 0], diff_bg_pts[:, 1], diff_bg_pts[:, 2],
            c="#e67e22" if not dark_theme else "#e67e22",
            s=point_size*1.5,
            alpha=0.65,
            linewidths=0, 
            depthshade=False
        )

        # --- Plot Object Points and Changed Points ---
        ax.scatter(
            obj_points_i[:, 0], obj_points_i[:, 1], obj_points_i[:, 2],
            c="#2ecc71",  # green
            s=point_size * 1.5,
            alpha=0.75,
            linewidths=0
        ).set_sort_zpos(0)

        diff_xyz = nudge_toward_camera(diff_points_i[:, :3], elev, azim, amount=diff_push)

        ax.scatter(
            diff_xyz[:, 0], diff_xyz[:, 1], diff_xyz[:, 2],
            c="#e67e22",
            s=point_size * 2,
            alpha=1.0,
            linewidths=0,
            depthshade=False
        )

        # --- Legend (use proxy artists for 3D scatter/lines) ---
        legend_elements = [
            # Line2D([0], [0], marker='o', color='none',
            #     markerfacecolor="#b0b0b0" if not dark_theme else "#8a8aa3",
            #     markersize=6, alpha=0.6, label='Scene Points'),
            Line2D([0], [0], marker='o', color='none',
                markerfacecolor="#e67e22", markersize=8, label='Changed points'),
            Line2D([0], [0], marker='o', color='none',
                markerfacecolor="#2ecc71", markersize=8, label='Object points'),
            # Line2D([0], [0], color="#2ecc71", linewidth=3, linestyle='-', label='GT box'),
        ]
        leg = ax.legend(handles=legend_elements, loc="lower center", fontsize=20, ncol=2)
        if dark_theme:
            leg.get_frame().set_facecolor('#2d2d44')
            leg.get_frame().set_edgecolor('white')
            for text in leg.get_texts():
                text.set_color('white')

        if not grid:
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_axis_off()
            fig.tight_layout()
            # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        else:
            fig.tight_layout()

        objects_dir = os.path.join(output_path, "adv_objects")
        os.makedirs(objects_dir, exist_ok=True)
        output_path_obj = os.path.join(output_path, f'adv_objects/{filename_stem}_obj_{i}.png')
        plt.savefig(output_path_obj, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
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
    compares advesarial with clean object
    """
    # gt box
    outer_box = sample.outer_boxes
    # clean/adv points per obj
    if adv:
        points_per_obj = sample.adv_points_per_obj
        pts = np.asarray(sample.adv_points.cpu())
        is_adv = "adv"
    else:
        points_per_obj = sample.points_per_obj
        pts = np.asarray(sample.points.cpu())
        is_adv = "clean"

    gt_box_corners = outer_box.corners.cpu()

    for i,(obj_p, gt_corner) in enumerate(zip(points_per_obj, gt_box_corners)):
        (xlim, ylim, zlim) = compute_dynamic_limits(
            obj_p[:, :3],
            gt_boxes_corners=gt_corner,
            pred_boxes_corners=gt_corner,
            percentiles=(5, 95),
            margin=-0.0,
            z_margin=-0.0,
            zoom = 0.7,
        )
        # --- Normalize shapes ---
        obj_points_i = np.asarray(obj_p.cpu())

        gt_box_corners = np.asarray(gt_corner)

        if gt_box_corners.shape == (3, 8):
            gt_box_for_draw = gt_box_corners
        else:
            gt_box_for_draw = gt_box_corners.T

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        if dark_theme:
            fig.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig, ax, is_3d=True)  # your existing helper

        # --- Boxes ---
        draw_box(ax, gt_box_for_draw, axes=[0, 1, 2], color="#2ecc71", linestyle="-", linewidth=1.0)

        # --- Limits: frame around OUTER box (plus margin) ---
        outer_corners_8x3 = gt_box_for_draw.T  # (8,3)
        mins = outer_corners_8x3.min(axis=0) - margin
        maxs = outer_corners_8x3.max(axis=0) + margin

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        set_axes_equal_3d(ax)

        # adjust view dynamically
        target = outer_corners_8x3.mean(axis=0)
        elev, azim = view_from_lidar(target, lidar_xyz=(0.0, 0.0, 0.0))
        ax.view_init(elev=elev, azim=azim)

        # choose an amount relative to box size (robust) to make sure that diff points are in foreground
        box_diag = np.linalg.norm(maxs - mins)
        diff_push = 0.01 * box_diag  # 1% of diagonal (tune 0.005–0.02)

        # --- Points: background object points ---
        # Remove background points inside GT box
        
        bg_pts = remove_points_inside_obb(pts[:, :3], outer_corners_8x3)

        ax.scatter(
            bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2],
            c="#b0b0b0" if not dark_theme else "#fcfafa",
            s=point_size,
            alpha=BACKGROUND_ALPHA,
            linewidths=0, 
            depthshade=False
        )

        # --- Plot Object Points and Changed Points ---
        ax.scatter(
            obj_points_i[:, 0], obj_points_i[:, 1], obj_points_i[:, 2],
            c="#2ecc71",  # green
            s=point_size * 2.5,
            alpha=0.75,
            linewidths=0
        ).set_sort_zpos(0)

        # # --- Legend (use proxy artists for 3D scatter/lines) ---
        # legend_elements = [
        #     Line2D([0], [0], marker='o', color='none',
        #         markerfacecolor="#b0b0b0" if not dark_theme else "#8a8aa3",
        #         markersize=6, alpha=0.6, label='Scene Points'),
        #     Line2D([0], [0], marker='o', color='none',
        #         markerfacecolor="#2ecc71", markersize=8, label='Object points'),
        #     Line2D([0], [0], color="#2ecc71", linewidth=3, linestyle='-', label='GT box'),
        # ]
        # leg = ax.legend(handles=legend_elements, loc="upper right", fontsize=20)
        # if dark_theme:
        #     leg.get_frame().set_facecolor('#2d2d44')
        #     leg.get_frame().set_edgecolor('white')
        #     for text in leg.get_texts():
        #         text.set_color('white')

        if not grid:
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_axis_off()
            fig.tight_layout()
            # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        else:
            fig.tight_layout()

        objects_dir = os.path.join(output_path, "adv_objects/sidebyside")
        os.makedirs(objects_dir, exist_ok=True)
        output_path_obj = os.path.join(output_path, f'adv_objects/sidebyside/{filename_stem}_obj_{i}_{is_adv}.png')
        plt.savefig(output_path_obj, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)


def nudge_toward_camera(points_xyz, elev_deg, azim_deg, amount=0.02):
    """
    Move points slightly toward the camera (in world coordinates).

    amount is in the same units as your point cloud.
    Use something like 0.5%–2% of the box diagonal.
    """
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    # Matplotlib's view direction convention (works well in practice):
    # unit vector pointing from origin toward the camera
    cam_dir = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev)
    ], dtype=float)

    # To move points "toward camera", shift along +cam_dir
    return points_xyz + amount * cam_dir


def remove_points_inside_obb(points, box_corners, eps=1e-6):
    """
    Remove points inside an oriented bounding box.

    points: (N,3)
    box_corners: (8,3) corners of box
    returns: filtered_points (M,3)
    """

    corners = np.asarray(box_corners)

    # --- 1. Compute box center ---
    center = corners.mean(axis=0)

    # --- 2. Compute box axes ---
    # Pick one corner as reference (corner 0)
    c0 = corners[0]

    # Find 3 closest corners to c0 -> these form the box edges
    dists = np.linalg.norm(corners - c0, axis=1)
    edge_indices = np.argsort(dists)[1:4]  # skip self

    edges = corners[edge_indices] - c0

    # Orthonormal basis
    axes = []
    lengths = []

    for e in edges:
        length = np.linalg.norm(e)
        axes.append(e / length)
        lengths.append(length)

    axes = np.stack(axes, axis=1)  # shape (3,3)
    half_lengths = np.array(lengths) / 2.0

    # --- 3. Transform points into box local frame ---
    rel_points = points - center
    local = rel_points @ axes  # projection

    # --- 4. Check inside box ---
    inside_mask = np.all(
        np.abs(local) <= (half_lengths + eps),
        axis=1
    )

    # Keep only outside points
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
    inner_box_corners = inner_box.corners.cpu()
    outer_box_corners = outer_box.corners.cpu()
    pts = np.asarray(points.cpu())
    for i,(obj_p, inn_pts,out_pts,inn_corner,out_corner) in enumerate(zip(obj_points,inner_points,outer_points, inner_box_corners, outer_box_corners)):
        (xlim, ylim, zlim) = compute_dynamic_limits(
            obj_p[:, :3],
            gt_boxes_corners=out_corner,
            pred_boxes_corners=inn_corner,
            percentiles=(0, 100),
            margin=margin,
            z_margin=margin,
        )

        # --- Normalize shapes ---
        obj_points_i = np.asarray(points.cpu())
        inner_points_i = np.asarray(inn_pts.cpu())
        outer_points_i = np.asarray(out_pts.cpu())

        inner_box_corners = np.asarray(inn_corner)
        outer_box_corners = np.asarray(out_corner)

        # If corners are (8,3) convert to (3,8) for corners.T usage, or adjust to your draw_box expectations
        # Your code uses draw_box(ax_3d, corners.T, axes=[0,1,2], ...)
        # so corners should be (8,3) -> corners.T becomes (3,8)
        if inner_box_corners.shape == (3, 8):
            inner_box_for_draw = inner_box_corners
        else:
            inner_box_for_draw = inner_box_corners.T

        if outer_box_corners.shape == (3, 8):
            outer_box_for_draw = outer_box_corners
        else:
            outer_box_for_draw = outer_box_corners.T

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        if dark_theme:
            fig.patch.set_facecolor('#1a1a2e')
            setup_dark_theme(fig, ax, is_3d=True)  # your existing helper

        # --- Points: background object points ---
        outer_corners_8x3 = outer_box_for_draw.T  # (8,3)

        bg_pts = remove_points_inside_obb(pts[:, :3], outer_corners_8x3)
        ax.scatter(
            bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2],
            c="#b0b0b0" if not dark_theme else "#fcfafa",
            s=point_size,
            alpha=BACKGROUND_ALPHA,
            linewidths=0
        )

        # --- Points per obj ---
        # ax.scatter(
        #     obj_points_i[:, 0], obj_points_i[:, 1], obj_points_i[:, 2],
        #     c="#b0b0b0" if not dark_theme else "#8a8aa3",
        #     s=point_size,
        #     alpha=0.25,
        #     linewidths=0
        # )

        # --- Inner / outer points: highlight ---
        ax.scatter(
            inner_points_i[:, 0], inner_points_i[:, 1], inner_points_i[:, 2],
            c="#2ecc71",  # green
            s=point_size * 2.5,
            alpha=0.9,
            linewidths=0
        )

        ax.scatter(
            outer_points_i[:, 0], outer_points_i[:, 1], outer_points_i[:, 2],
            c="#e67e22",  # orange
            s=point_size * 2.5,
            alpha=0.9,
            linewidths=0
        )

        # --- Boxes ---
        draw_box(ax, inner_box_for_draw, axes=[0, 1, 2], color="#2ecc71", linestyle="-", linewidth=1.0)
        draw_box(ax, outer_box_for_draw, axes=[0, 1, 2], color="#e67e22", linestyle="-", linewidth=1.0)

        # adjust view dynamically
        target = outer_corners_8x3.mean(axis=0)
        elev, azim = view_from_lidar(target, lidar_xyz=(0.0, 0.0, 0.0))
        # ax.view_init(elev=elev, azim=azim) # Facing camera
        ax.view_init(elev=view[0], azim=view[1]) # set angle

        # --- Limits: frame around OUTER box (plus margin) ---
        ax.set_xlim3d(*xlim)
        ax.set_ylim3d(*ylim)
        ax.set_zlim3d(*zlim)
        set_axes_equal_3d(ax)

        # ax.set_title("Object points with inner/outer split + inner/outer boxes", fontsize=12)
        # ax.set_xlabel("X (m)")
        # ax.set_ylabel("Y (m)")
        # ax.set_zlabel("Z (m)")

        # --- Legend (use proxy artists for 3D scatter/lines) ---
        legend_elements = [
            # Line2D([0], [0], marker='o', color='none',
            #     markerfacecolor="#b0b0b0" if not dark_theme else "#8a8aa3",
            #     markersize=6, alpha=0.6, label='Scene Points'),
            # Line2D([0], [0], marker='o', color='none',
            #     markerfacecolor="#2ecc71", markersize=8, label='Inner points'),
            # Line2D([0], [0], marker='o', color='none',
            #     markerfacecolor="#e67e22", markersize=8, label='Outer points'),
            Line2D([0], [0], color="#2ecc71", linewidth=3, linestyle='-',  label='Inner box'),
            Line2D([0], [0], color="#e67e22", linewidth=3, linestyle='-', label='Outer box'),
        ]
        leg = ax.legend(
            handles=legend_elements, 
            loc="lower center", 
            fontsize=35,
            ncol=2
            )
        if dark_theme:
            leg.get_frame().set_facecolor('#2d2d44')
            leg.get_frame().set_edgecolor('white')
            for text in leg.get_texts():
                text.set_color('white')

        if not grid:
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_axis_off()
            fig.tight_layout()
            # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        else:
            fig.tight_layout()

        objects_dir = os.path.join(output_path, "objects")
        os.makedirs(objects_dir, exist_ok=True)
        output_path_obj = os.path.join(output_path, f'objects/{filename_stem}_obj_{i}.png')
        plt.savefig(output_path_obj, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

def set_axes_equal_3d(ax):
    """Make 3D plot axes have equal scale so boxes look like boxes."""
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
    Returns (elev, azim) for matplotlib's ax.view_init such that the camera
    looks toward target_xyz with a direction consistent with lidar_xyz -> target_xyz.
    """
    dx, dy, dz = (np.asarray(target_xyz, dtype=float) - np.asarray(lidar_xyz, dtype=float))

    # azim: angle in XY plane (0 = +X, 90 = +Y)
    azim = np.degrees(np.arctan2(dy, dx)) + 180

    # elev: angle above XY plane
    horiz = np.hypot(dx, dy)
    elev = np.degrees(np.arctan2(dz, horiz)) + 15

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
    adv_point_cloud = sample.diff_pc_list(point_cloud, adv_point_cloud)
    gt_boxes_corners = sample.gt_boxes.corners
    gt_labels = sample.gt_labels

    pred_instances = sample.result["pts_bbox"]
    scores = np.array([s.item() for s in sample.result["pts_bbox"]["scores_3d"]])
    mask = scores > show_thr
    pred_boxes_corners = pred_instances["boxes_3d"][mask].corners
    pred_labels = pred_instances["labels_3d"][mask]

    adv_pred_instances = sample.adv_result["pts_bbox"]
    adv_scores = np.array([s.item() for s in sample.adv_result["pts_bbox"]["scores_3d"]])
    mask = adv_scores > show_thr
    pred_boxes_corners_adv = adv_pred_instances["boxes_3d"][mask].corners
    pred_labels_adv = adv_pred_instances["labels_3d"][mask]

    # Convert labels
    LABEL_MAP = np.array(CLASS_NAMES[dataset])

    gt_ids = np.asarray(gt_labels).astype(int).reshape(-1)
    pred_ids = np.asarray(pred_labels).astype(int).reshape(-1)
    pred_ids_adv = np.asarray(pred_labels_adv).astype(int).reshape(-1)

    gt_labels_string = LABEL_MAP[gt_ids]
    pred_labels_string = LABEL_MAP[pred_ids]
    pred_labels_string_adv = LABEL_MAP[pred_ids_adv]

    fig_bev = plt.figure(figsize=bev_figsize, dpi=bev_dpi)
    ax_bev = fig_bev.add_subplot(111)

    if dark_theme:
        fig_bev.patch.set_facecolor('#1a1a2e')
        setup_dark_theme(fig_bev, ax_bev, is_3d=False)


    # 2. Subsample points
    points_step = int(1. / points_keep_ratio) if points_keep_ratio > 0 else 1
    velo_range = range(0, point_cloud.shape[0], points_step)
    velo_frame_clean = point_cloud[velo_range, :]

    adv_velo_range = range(0, adv_point_cloud.shape[0], points_step)
    velo_frame_adv = adv_point_cloud[adv_velo_range, :]

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
        show_only_present=True,   # shows only classes that appear in this sample
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
    # -----------------------------
    # 1) Scatter BOTH point clouds
    # -----------------------------
    # Clean points (use your existing color logic)
    sc_clean = ax_bev.scatter(
        velo_frame_clean[:, 0],
        velo_frame_clean[:, 1],
        c="#fcfafa",            # White
        s=point_size,
        alpha=1,
        linewidths=0,
        label="Points (clean)"
    )

    # Adv points (pick one of the two approaches)
    # (A) single color for adv points (recommended for clarity)
    adv_point_color = "#ff0000" if not dark_theme else "#e67e22"
    sc_adv = ax_bev.scatter(
        velo_frame_adv[:, 0],
        velo_frame_adv[:, 1],
        c=adv_point_color,
        s=point_size*4,
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

    # -----------------------------
    # 2) Draw boxes: GT, clean pred, adv pred
    # -----------------------------

    # GT boxes (solid, thick)
    for corners, label in zip(gt_boxes_corners, gt_labels_string):
        if label in BOX_COLORS:
            draw_box(
                ax_bev, corners.T, axes=[0, 1],
                color=BOX_COLORS[label],
                linestyle='-',
                linewidth=3.0
            )

    # Clean prediction boxes (dashed)
    for corners, label in zip(pred_boxes_corners, pred_labels_string):
        if label in BOX_COLORS:
            draw_box(
                ax_bev, corners.T, axes=[0, 1],
                color=BOX_COLORS[label],
                linestyle='--',
                linewidth=2.5
            )

    # Adv prediction boxes (dash-dot or dotted)
    for corners, label in zip(pred_boxes_corners_adv, pred_labels_string_adv):
        if label in BOX_COLORS:
            draw_box(
                ax_bev, corners.T, axes=[0, 1],
                color=BOX_COLORS[label],
                linestyle='-.',         # or ':' if you want more contrast
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
    plt.savefig(output_path_bev, dpi=bev_dpi, facecolor=fig_bev.get_facecolor(), bbox_inches='tight')
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

    Args:
        dataset: dataset key used in CLASS_NAMES
        CLASS_NAMES: dict/list mapping dataset -> list of class names
        box_colors: dict mapping canonical class-name keys -> color (e.g. {'Car': ..., ...})
        gt_labels_string / pred_labels_string: arrays/lists of per-box class names (strings)
        show_only_present: if True, only include classes that appear in GT or Pred for this frame
        include_pred: include dashed pred legend entries
        gt_style: (linestyle, linewidth) for GT
        pred_style: (linestyle, linewidth) for Pred

    Returns:
        legend_elements: list of Line2D handles
    """
    # Dataset class names (as strings)
    dataset_classes = [str(x) for x in CLASS_NAMES[dataset]]

    # Normalize color keys (e.g., 'Car' -> 'car')
    keymap = {str(k).strip().lower(): k for k in box_colors.keys()}

    # Which classes are present in this frame (optional)
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

        # skip if class has no color configured
        if cname_l not in keymap:
            continue

        # skip if not present in this frame and user asked for only-present
        if present is not None and cname_l not in present:
            continue

        canonical_key = keymap[cname_l]
        color = box_colors[canonical_key]

        # GT handle
        legend_elements.append(
            Line2D([0], [0], color=color, lw=gt_lw, linestyle=gt_ls, label=f"{cname} (GT)")
        )

        # Pred handle
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
    percentiles=(1.0, 99.0),   # robust against outliers
    margin=2.0,                # meters
    z_margin=1.0,
    dataset_clip=None,         # optional dict: {"x":(min,max), "y":..., "z":...}
    zoom = 1.0,
):
    """
    Compute dynamic axis limits from points (+ optional boxes), robustly.

    points_xyz: (N, >=3)
    gt_boxes_corners/pred_boxes_corners: (M, 8, 3) arrays (or lists) in same coord frame as points
    percentiles: use robust percentile range instead of raw min/max
    dataset_clip: optional hard clip to keep views comparable across samples
    """
    xs = [points_xyz[:, 0].cpu()]
    ys = [points_xyz[:, 1].cpu()]
    zs = [points_xyz[:, 2].cpu()]

    def add_boxes(boxes):
        if boxes is None:
            return
        b = np.asarray(boxes)
        if b.size == 0:
            return
        # expect (M, 8, 3) OR (M, 3, 8) etc; normalize to (..., 3)
        b = b.reshape(-1, 3)
        xs.append(b[:, 0]); ys.append(b[:, 1]); zs.append(b[:, 2])

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
    colormap = plt.cm.get_cmap(cmap)
    return colormap(norm(distances)), distances, 'Distance (m)'


def get_point_colors_by_height(points, cmap='viridis'):
    """Color points by Z-height - good for seeing ground vs objects."""
    z_vals = points[:, 2]
    norm = Normalize(vmin=z_vals.min(), vmax=z_vals.max())
    colormap = plt.cm.get_cmap(cmap)
    return colormap(norm(z_vals)), z_vals, 'Height (m)'


def get_point_colors_by_intensity(points, cmap='hot'):
    """Color points by intensity (4th channel if available)."""
    if points.shape[1] >= 4:
        intensity = points[:, 3]
    else:
        intensity = np.linalg.norm(points[:, :3], axis=1)  # fallback to depth
    norm = Normalize(vmin=intensity.min(), vmax=intensity.max())
    colormap = plt.cm.get_cmap(cmap)
    return colormap(norm(intensity)), intensity, 'Intensity'


def get_point_colors_by_density(points, radius=0.5, cmap='coolwarm'):
    """Color by local point density - highlights adversarial clusters."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, :3])
    density = np.array([len(tree.query_ball_point(p, radius)) for p in points[:, :3]])
    norm = Normalize(vmin=density.min(), vmax=density.max())
    colormap = plt.cm.get_cmap(cmap)
    return colormap(norm(density)), density, 'Local Density'


def get_point_colors(points, mode='depth'):
    """Get colors based on selected mode."""
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
    vertices: (3, 8) numpy array
    """
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=values.min(), vmax=values.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(label, fontsize=10)
    
    if dark_theme:
        cbar.set_label(label, color='white', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')


if __name__ == '__main__':
    main()