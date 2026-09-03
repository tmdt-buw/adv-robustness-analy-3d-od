import pandas as pd
from os import path as osp
from sample import Sample
import sqlite3
import pickle
import numpy as np
import torch
import time
import os, glob, re

from utils import iter_results_db

class SummaryTable(object):
    
    def __init__(self, path, save_path="eval.db", compute_nds = False, dist_thresh = 0.2, fp_thresh = 0.15, innout_thresh = 0.8, match_method="dist", full=False):
        conn = sqlite3.connect(save_path)
        cur = conn.cursor()

        # Performance + safety
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=OFF;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA foreign_keys=ON;")

        # ---------- samples ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT NOT NULL UNIQUE,

            pct_change REAL,
            num_gt INTEGER,
            num_missed_gt INTEGER,
            adv_num_missed_gt INTEGER,
            distinct_classes INTEGER,

            num_fp INTEGER,
            num_tp INTEGER,
            adv_num_fp INTEGER,
            adv_num_tp INTEGER,

            yaw_error_change REAL,
            scale_error_change REAL,
            adv_yaw_error_change REAL,
            adv_scale_error_change REAL,

            asr REAL,
            ddr REAL,
            chamfer_dist REAL,

            recall REAL,
            precision REAL,
            adv_recall REAL,
            adv_precision REAL,

            score_01 REAL,
            score_05 REAL,
            adv_score_01 REAL,
            adv_score_05 REAL
        );
        """)

        # ---------- boxes ----------
        cur.execute("""
        CREATE TABLE IF NOT EXISTS boxes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            sample_id TEXT NOT NULL,
            gt_box_id INTEGER NOT NULL,

            class INTEGER,
            pred_class INTEGER,
            adv_pred_class INTEGER,

            conf REAL,
            adv_conf REAL,

            iou REAL,
            adv_iou REAL,

            attack_success INTEGER,
            distance_car REAL,

            num_points INTEGER,
            adv_num_points INTEGER,

            pct_obj_points_change REAL,
            pct_inner_points_change REAL,
            pct_outer_points_change REAL,
            amt_inner_points INTEGER,
            amt_outer_points INTEGER,

            yaw_err REAL,
            adv_yaw_err REAL,
            trans_err REAL,
            adv_trans_err REAL,
            scale_err REAL,
            adv_scale_err REAL
        );
        """)

        if full:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS pred_boxes (
                sample_id TEXT NOT NULL,
                is_adv    INTEGER NOT NULL,
                pred_id   INTEGER NOT NULL,     -- your "rank"/index after sorting by score
                class_id  INTEGER NOT NULL,
                score     REAL NOT NULL,
                PRIMARY KEY (sample_id, is_adv, pred_id)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS pred_gt_candidates (
                sample_id TEXT NOT NULL,
                is_adv    INTEGER NOT NULL,
                pred_id   INTEGER NOT NULL,
                gt_id     INTEGER NOT NULL,
                dist_xy   REAL,                 -- center distance (xy)
                iou_3d    REAL,                 -- oriented 3D IoU (nullable if you skipped compute)
                vel_err  REAL,
                scale_err  REAL,
                trans_err  REAL,
                orient_err  REAL,
                attr_err  REAL,
                PRIMARY KEY (sample_id, is_adv, pred_id, gt_id)
            );
            """)

        # ---------- indexes ----------
        cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_sample_id ON samples(sample_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_boxes_sample_id ON boxes(sample_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_boxes_class ON boxes(class)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_boxes_attack_success ON boxes(attack_success)")
        if full:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_class_score ON pred_boxes(class_id, score DESC, is_adv)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cand_pred ON pred_gt_candidates(sample_id, is_adv, pred_id)")

        conn.commit()

        # ---------- resume support ----------
        cur.execute("SELECT sample_id FROM samples;")
        done_samples = set(r[0] for r in cur.fetchall())
        print(f"Resuming: {len(done_samples)} samples already done")

        counter = 0
        skipped = 0
        # print("Starting loop!")
        for res in iter_results_db(path):
            sample_id = res["name"]

            # No need for resume support currently, it still needs to read in the entire file
            if sample_id in done_samples:
                skipped += 1
                continue

            sample = Sample(
                res,
                dist_thresh=dist_thresh,
                fp_thresh=fp_thresh,
                match_method=match_method,
                innout_thresh = innout_thresh,
            )

            # # Debug stop
            # print("Remove line 161! Stopped after first sample for debugging!!!!!")
            # break

            try:
                # ---------- insert sample ----------
                cur.execute("""
                INSERT INTO samples (
                    sample_id,
                    pct_change, num_gt, num_missed_gt, adv_num_missed_gt, distinct_classes,
                    num_fp, num_tp, adv_num_fp, adv_num_tp,
                    yaw_error_change, scale_error_change,
                    adv_yaw_error_change, adv_scale_error_change,
                    asr, ddr, chamfer_dist,
                    recall, precision, adv_recall, adv_precision,
                    score_01, score_05, adv_score_01, adv_score_05
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                self.to_sqlite(sample.table_entry["sample_id"]),
                self.to_sqlite(sample.table_entry["%_change"]),
                self.to_sqlite(sample.table_entry["num_gt"]),
                self.to_sqlite(sample.table_entry["num_missed_gt"]),
                self.to_sqlite(sample.table_entry["adv_num_missed_gt"]),
                self.to_sqlite(sample.table_entry["distinct_classes"]),
                self.to_sqlite(sample.table_entry["num_fp"]),
                self.to_sqlite(sample.table_entry["num_tp"]),
                self.to_sqlite(sample.table_entry["adv_num_fp"]),
                self.to_sqlite(sample.table_entry["adv_num_tp"]),
                self.to_sqlite(sample.table_entry["yaw_error_change"]),
                self.to_sqlite(sample.table_entry["scale_error_change"]),
                self.to_sqlite(sample.table_entry["adv_yaw_error_change"]),
                self.to_sqlite(sample.table_entry["adv_scale_error_change"]),
                self.to_sqlite(sample.table_entry["asr"]),
                self.to_sqlite(sample.table_entry["ddr"]),
                self.to_sqlite(sample.table_entry["chamfer_dist"]),
                self.to_sqlite(sample.table_entry["recall"]),
                self.to_sqlite(sample.table_entry["precision"]),
                self.to_sqlite(sample.table_entry["adv_recall"]),
                self.to_sqlite(sample.table_entry["adv_precision"]),
                self.to_sqlite(sample.table_entry["score@0.1"]),
                self.to_sqlite(sample.table_entry["score@0.5"]),
                self.to_sqlite(sample.table_entry["adv_score@0.1"]),
                self.to_sqlite(sample.table_entry["adv_score@0.5"]),
                ))

            except sqlite3.IntegrityError:
                skipped += 1
                continue

            # ---------- insert boxes ----------
            cur.executemany("""
                INSERT INTO boxes (
                    sample_id, gt_box_id,
                    class, pred_class, adv_pred_class,
                    conf, adv_conf,
                    iou, adv_iou,
                    attack_success,
                    distance_car,
                    num_points, adv_num_points,
                    pct_obj_points_change,
                    pct_inner_points_change,
                    pct_outer_points_change,
                    amt_inner_points,
                    amt_outer_points,
                    yaw_err, adv_yaw_err,
                    trans_err, adv_trans_err,
                    scale_err, adv_scale_err
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, [
                (
                    self.to_sqlite(b["sample_id"]),
                    self.to_sqlite(b["gt_box_id"]),
                    self.to_sqlite(b["class"]),
                    self.to_sqlite(b["pred_class"]),
                    self.to_sqlite(b["adv_pred_class"]),
                    self.to_sqlite(b["conf"]),
                    self.to_sqlite(b["adv_conf"]),
                    self.to_sqlite(b["iou"]),
                    self.to_sqlite(b["adv_iou"]),
                    int(b["attack_success"]),  # ensure boolean is int
                    self.to_sqlite(b["distance_car"]),
                    self.to_sqlite(b["num_points"]),
                    self.to_sqlite(b["adv_num_points"]),
                    self.to_sqlite(b["%_obj_points_change"]),
                    self.to_sqlite(b["%_inner_points_change"]),
                    self.to_sqlite(b["%_outer_points_change"]),
                    self.to_sqlite(b["amt_inner_points"]),
                    self.to_sqlite(b["amt_outer_points"]),
                    self.to_sqlite(b["yaw_err"]),
                    self.to_sqlite(b["adv_yaw_err"]),
                    self.to_sqlite(b["trans_err"]),
                    self.to_sqlite(b["adv_trans_err"]),
                    self.to_sqlite(b["scale_err"]),
                    self.to_sqlite(b["adv_scale_err"]),
                )
                for b in sample.box_table
            ])

            # mAP calc tables
            if full:
                rows_pred_all = []
                rows_cand_all = []

                # clean
                rows_pred_all.extend(sample.map_data["clean"])
                rows_cand_all.extend(sample.candidates["clean"])

                # adv
                rows_pred_all.extend(sample.map_data["adv"])
                rows_cand_all.extend(sample.candidates["adv"])

                cur.executemany("""
                    INSERT OR IGNORE INTO pred_boxes (
                        sample_id, is_adv, pred_id, class_id, score
                    ) VALUES (?,?,?,?,?)
                """, rows_pred_all)

                cur.executemany("""
                    INSERT OR IGNORE INTO pred_gt_candidates (
                        sample_id, is_adv, pred_id, gt_id, dist_xy, iou_3d, vel_err, scale_err, trans_err, orient_err, attr_err
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, rows_cand_all)


            counter += 1
            if counter % 100 == 0:
                conn.commit()
                print(f"Processed {counter}, skipped {skipped}")

            del sample

        # ---------- indexes ----------
        cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_sample_id ON samples(sample_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_boxes_sample_id ON boxes(sample_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_boxes_class ON boxes(class)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_boxes_attack_success ON boxes(attack_success)")
        conn.commit()

        # ---------- final debug ----------
        print(f"=" * 60)
        print(f"Finished Table Summary")
        print(f"=" * 60)
        cur.execute("SELECT COUNT(*) FROM samples;")
        print("Final samples:", cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM boxes;")
        print("Final boxes:", cur.fetchone()[0])
        if full:
            cur.execute("SELECT COUNT(*) FROM pred_boxes;")
            print("Final pred_boxes:", cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM pred_gt_candidates;")
            print("Final pred_box_candidates:", cur.fetchone()[0])
        print(f"=" * 60)
        
        conn.close()


    def to_sqlite(self, val):
        if val is None:
            return -1
        if isinstance(val, (int, float, str)):
            return val
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return float(val.item())  # convert scalar tensor to float
            else:
                raise TypeError(f"Cannot store multi-element tensor in SQLite: {val}")
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.bool_):
            return int(val)
        raise TypeError(f"Unsupported type for SQLite: {type(val)} -> {val}")



