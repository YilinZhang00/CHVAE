# dataset.py
import os
import random
from typing import Optional, Tuple, Dict, Callable, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# ---------------------- I/O: DICOM -> Tensor ----------------------
def read_dicom_as_tensor(
    path: str,
    img_size: int = 128,
    window: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Read a single DICOM and return a tensor of shape [1, H, W], dtype float32, ranged in [0, 1].
    Steps:
      1) dcmread + apply_voi_lut
      2) Apply RescaleSlope/RescaleIntercept
      3) Invert if MONOCHROME1
      4) Normalize to [0,1] (by window or per-image min/max)
      5) Resize to (img_size, img_size) via bilinear interpolation
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DICOM not found: {path}")

    ds = pydicom.dcmread(path, force=True)

    # Raw pixels -> float
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)

    # RescaleSlope/Intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Invert MONOCHROME1
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = np.max(arr) - arr

    img = torch.from_numpy(arr).to(dtype=torch.float32)

    # Normalize
    if window is not None:
        wmin, wmax = float(window[0]), float(window[1])
        if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
            raise ValueError(f"Invalid window range: {window}")
        img = torch.clamp(img, min=wmin, max=wmax)
        img = (img - wmin) / (wmax - wmin + 1e-6)
    else:
        imin = torch.min(img)
        imax = torch.max(img)
        if (imax - imin) < 1e-6:
            img = torch.zeros_like(img)
        else:
            img = (img - imin) / (imax - imin + 1e-6)

    # Safety
    img = torch.clamp(img, 0.0, 1.0).nan_to_num(0.0)

    # [H,W] -> [1,1,H,W] for interpolation
    img = img.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    img = F.interpolate(img, size=(img_size, img_size), mode="bilinear", align_corners=False)
    img = img.squeeze(0)  # [1,H,W]
    return img.contiguous()


# ---------------------- Helpers: coerce metadata to numeric ----------------------
def _coerce_numeric(series: pd.Series, name: str,
                    categorical_map: Optional[Dict] = None) -> pd.Series:
    """
    Convert mixed/string columns to numeric.
    - If categorical_map is provided, use it.
    - Else try to_numeric; for 'sex' with common strings do {male/m -> 1.0, female/f -> 0.0}.
    """
    if categorical_map is not None:
        return series.map(categorical_map)

    out = pd.to_numeric(series, errors="coerce")

    if out.isna().any() and series.dtype == object:
        lower = series.astype(str).str.lower()
        if set(lower.dropna().unique()) <= {"male", "m", "female", "f"}:
            mapped = lower.map({"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0})
            return mapped

    return out


# ---------------------- Dataset ----------------------
class DXAEidDicomMetaDataset(Dataset):
    """
    Directory layout:
      root_dir/
        └─ <eid>/
            ├─ <eid>_instance_2.dcm      # required
            └─ <eid>_instance_3.dcm      # optional

    The metadata TSV contains per-EID covariates.

    Returned sample dict:
      {
        "eid": <str> (if return_eid=True),
        "image": Tensor[1,H,W] in [0,1],                         # instance 2 image (required)
        "sex": Tensor([]),
        "age": Tensor([]),
        "physical_activity": Tensor([]),
        "standing_height": Tensor([]),
        "l14_width": Tensor([]),
        "l14_height": Tensor([]),
        "l14_area": Tensor([]),
        "weight": Tensor([]),
        "path": <str>,                                           # instance_2 DICOM path

        # Optional external validation from instance 3 (if available):
        "inst3_image": Tensor[1,H,W] in [0,1],
        "inst3_age": Tensor([]), "inst3_l14_width": Tensor([]),
        "inst3_l14_height": Tensor([]), "inst3_l14_area": Tensor([]),
        "inst3_weight": Tensor([]), "inst3_standing_height": Tensor([]),
        "inst3_physical_activity": Tensor([]), "inst3_sex": Tensor([]),
        "has_inst3_image": Tensor([])  # 1.0 if inst3 image present else 0.0
      }
    """
    REQUIRED_KEYS: List[str] = [
        "sex", "age", "physical_activity", "standing_height",
        "l14_width", "l14_height", "l14_area", "weight"
    ]
    OPTIONAL_INST3_KEYS: List[str] = [
        "sex", "age", "physical_activity", "standing_height",
        "l14_width", "l14_height", "l14_area", "weight"
    ]

    def __init__(
        self,
        root_dir: str,
        metadata_tsv: str,
        column_map: Optional[Dict[str, str]] = None,
        img_size: int = 128,
        window: Optional[Tuple[float, float]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropna: bool = True,
        categorical_maps: Optional[Dict[str, Dict]] = None,
        eid_column: Optional[str] = None,
        strict_files: bool = True,
        return_eid: bool = True,
    ):
        # Normalize paths
        root_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(root_dir)))
        metadata_tsv = os.path.abspath(os.path.expanduser(os.path.expandvars(metadata_tsv)))
        if not os.path.isfile(metadata_tsv):
            raise FileNotFoundError(f"metadata_tsv not found: {metadata_tsv}")

        self.root_dir = root_dir
        self.img_size = int(img_size)
        self.window = window
        self.transform = transform
        self.dropna = dropna
        self.categorical_maps = categorical_maps or {}
        self.return_eid = return_eid
        self.strict_files = strict_files

        # Read TSV
        try:
            df = pd.read_csv(metadata_tsv, sep="\t")
        except Exception:
            df = pd.read_csv(metadata_tsv)

        # Column name resolution
        column_map = dict(column_map or {})
        if "eid" not in column_map:
            column_map["eid"] = eid_column or "participant.eid"

        # Aliases for instance-2 columns (required)
        ALIASES_I2: Dict[str, List[str]] = {
            "sex": ["Sex", "sex", "Gender", "gender"],
            "age": ["Age at DXA (instance 2)", "Age (instance 2)", "Age"],
            "physical_activity": ["Physical activity (instance 2)", "Physical activity"],
            "standing_height": [
                "Standing height (instance 2)", "Standing height (instance 0)",
                "Standing height (instance 3)", "Standing height"
            ],
            "l14_width":  ["L1-L4 average width (instance 2)", "L1–L4 average width (instance 2)", "L1-L4 width (instance 2)"],
            "l14_height": ["L1-L4 average height (instance 2)", "L1–L4 average height (instance 2)", "L1-L4 height (instance 2)"],
            "l14_area":   ["L1-L4 area (instance 2)", "L1–L4 area (instance 2)", "L1-L4 area (instance 2)"],
            "weight": ["Weight (instance 2)", "Weight"],
        }

        # Aliases for instance-3 columns (optional)
        ALIASES_I3: Dict[str, List[str]] = {
            "sex": ["Sex (instance 3)", "Sex3", "sex3", "Gender (instance 3)"],
            "age": ["Age at DXA (instance 3)", "Age (instance 3)"],
            "physical_activity": ["Physical activity (instance 3)"],
            "standing_height": ["Standing height (instance 3)"],
            "l14_width":  ["L1-L4 average width (instance 3)", "L1–L4 average width (instance 3)", "L1-L4 width (instance 3)"],
            "l14_height": ["L1-L4 average height (instance 3)", "L1–L4 average height (instance 3)", "L1-L4 height (instance 3)"],
            "l14_area":   ["L1-L4 area (instance 3)", "L1–L4 area (instance 3)", "L1-L4 area (instance 3)"],
            "weight": ["Weight (instance 3)"],
        }

        # Resolve EID column
        eid_col = column_map["eid"]
        if eid_col not in df.columns:
            for cand in ("participant.eid", "eid", "EID", "Id", "ID"):
                if cand in df.columns:
                    eid_col = cand
                    column_map["eid"] = cand
                    break
            else:
                raise KeyError(f"EID column '{eid_col}' not found. Available: {list(df.columns)}")

        def _pick_or_alias(std_key: str, aliases: Dict[str, List[str]]) -> str:
            # explicit mapping first
            if std_key in column_map and column_map[std_key] in df.columns:
                return column_map[std_key]
            # try aliases
            for cand in aliases.get(std_key, []):
                if cand in df.columns:
                    column_map[std_key] = cand
                    return cand
            # fall back to std_key
            if std_key in df.columns:
                column_map[std_key] = std_key
                return std_key
            raise KeyError(
                f"Column for '{std_key}' not found. "
                f"Tried: {column_map.get(std_key, std_key)!r} and aliases {aliases.get(std_key, [])}. "
                f"Available columns: {list(df.columns)}"
            )

        def _pick_optional(std_key: str, aliases: Dict[str, List[str]]) -> Optional[str]:
            # explicit mapping like 'inst3_age' in column_map
            cm_key = f"inst3_{std_key}"
            if cm_key in column_map and column_map[cm_key] in df.columns:
                return column_map[cm_key]
            for cand in aliases.get(std_key, []):
                if cand in df.columns:
                    column_map[cm_key] = cand
                    return cand
            return None

        # Resolve required instance-2 columns
        resolved_i2 = {"eid": eid_col}
        for k in self.REQUIRED_KEYS:
            resolved_i2[k] = _pick_or_alias(k, ALIASES_I2)

        # Resolve optional instance-3 columns
        resolved_i3: Dict[str, Optional[str]] = {}
        for k in self.OPTIONAL_INST3_KEYS:
            resolved_i3[k] = _pick_optional(k, ALIASES_I3)

        # Select and rename i2 columns
        use_cols = [resolved_i2["eid"]] + [resolved_i2[k] for k in self.REQUIRED_KEYS]
        sub = df[use_cols].copy()
        rename_map = {resolved_i2["eid"]: "eid"}
        rename_map.update({resolved_i2[k]: k for k in self.REQUIRED_KEYS})
        sub = sub.rename(columns=rename_map)
        sub["eid"] = sub["eid"].astype(str)

        # Coerce required columns to numeric
        for k in self.REQUIRED_KEYS:
            sub[k] = _coerce_numeric(sub[k], k, self.categorical_maps.get(k))

        # Attach optional instance-3 columns if present
        for k, col in resolved_i3.items():
            if col is not None:
                series = _coerce_numeric(df[col], k, self.categorical_maps.get(k))
                sub[f"inst3_{k}"] = series

        # Drop rows with missing required cols
        if self.dropna:
            sub = sub.dropna(subset=self.REQUIRED_KEYS + ["eid"]).copy()

        # Keep only EIDs with instance_2.dcm
        def _has_dcm_i2(eid: str) -> bool:
            dcm_path = os.path.join(self.root_dir, eid, f"{eid}_instance_2.dcm")
            if os.path.isfile(dcm_path):
                return True
            if self.strict_files:
                return False
            ddir = os.path.join(self.root_dir, eid)
            if not os.path.isdir(ddir):
                return False
            for f in os.listdir(ddir):
                if f.endswith(".dcm") and "instance_2" in f:
                    return True
            return False

        sub = sub[sub["eid"].apply(_has_dcm_i2)].reset_index(drop=True)
        if len(sub) == 0:
            raise RuntimeError("No valid EIDs with both metadata and instance_2.dcm were found.")

        self.meta = sub
        self.eids = sub["eid"].tolist()

    def __len__(self) -> int:
        return len(self.eids)

    def _resolve_dcm_path(self, eid: str, instance: int = 2) -> str:
        """
        Resolve DICOM path for a given EID and instance number (2 or 3).
        For instance=2, existence is required by dataset construction; for instance=3 it is optional.
        """
        suffix = f"{eid}_instance_{instance}.dcm"
        p = os.path.join(self.root_dir, eid, suffix)
        if os.path.isfile(p):
            return p
        ddir = os.path.join(self.root_dir, eid)
        if not os.path.isdir(ddir):
            raise FileNotFoundError(f"DICOM dir not found for EID {eid}")
        cands = [f for f in os.listdir(ddir) if f.endswith(".dcm") and f"instance_{instance}" in f]
        if cands:
            return os.path.join(ddir, cands[0])
        if instance == 2:
            raise FileNotFoundError(f"instance_2 DICOM not found for EID {eid}")
        else:
            raise FileNotFoundError(f"instance_{instance} DICOM not found for EID {eid}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        eid = self.eids[idx]
        row = self.meta.iloc[idx]

        # Instance 2 image (required)
        dcm_path_i2 = self._resolve_dcm_path(eid, instance=2)
        img_i2 = read_dicom_as_tensor(dcm_path_i2, self.img_size, self.window)  # [1,H,W]
        if self.transform is not None:
            img_i2 = self.transform(img_i2)  # transform expects/returns [1,H,W]

        sample: Dict[str, Any] = {
            "image": img_i2,
            "sex": torch.tensor(row["sex"], dtype=torch.float32),
            "age": torch.tensor(row["age"], dtype=torch.float32),
            "physical_activity": torch.tensor(row["physical_activity"], dtype=torch.float32),
            "standing_height": torch.tensor(row["standing_height"], dtype=torch.float32),
            "l14_width": torch.tensor(row["l14_width"], dtype=torch.float32),
            "l14_height": torch.tensor(row["l14_height"], dtype=torch.float32),
            "l14_area": torch.tensor(row["l14_area"], dtype=torch.float32),
            "weight": torch.tensor(row["weight"], dtype=torch.float32),
            "path": dcm_path_i2,
        }
        if self.return_eid:
            sample["eid"] = eid

        # Optional: instance 3 image
        has_i3 = 0.0
        dcm_path_i3 = os.path.join(self.root_dir, eid, f"{eid}_instance_3.dcm")
        if os.path.isfile(dcm_path_i3):
            try:
                img_i3 = read_dicom_as_tensor(dcm_path_i3, self.img_size, self.window)
                if self.transform is not None:
                    img_i3 = self.transform(img_i3)
                sample["inst3_image"] = img_i3
                has_i3 = 1.0
            except Exception:
                # Keep training robust when a few DICOMs are corrupted
                pass

        # Optional: instance 3 metadata columns (only attach if present and not NaN)
        for k in self.OPTIONAL_INST3_KEYS:
            col = f"inst3_{k}"
            if col in self.meta.columns and pd.notna(row.get(col, np.nan)):
                try:
                    sample[col] = torch.tensor(float(row[col]), dtype=torch.float32)
                except Exception:
                    pass

        # ---- standardize keys across samples (important for DataLoader collation)
        if "inst3_image" not in sample:
            sample["inst3_image"] = sample["image"].clone().zero_()  # zero placeholder
        sample["has_inst3_image"] = torch.tensor(has_i3, dtype=torch.float32)

        for k in self.OPTIONAL_INST3_KEYS:
            col = f"inst3_{k}"
            if col not in sample:
                sample[col] = torch.tensor(float("nan"), dtype=torch.float32)

        return sample


# ---------------------- Split builders ----------------------
def build_splits_and_datasets(
    root_dir: str,
    metadata_tsv: str,
    column_map: Optional[Dict[str, str]] = None,
    img_size: int = 128,
    window: Optional[Tuple[float, float]] = None,
    test_size: float = 0.2,
    val_size: float = 0.5,
    seed: int = 42,
    train_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    val_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    test_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    *,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # backward-compatible
    dropna: bool = True,
    categorical_maps: Optional[Dict[str, Dict]] = None,
    eid_column: Optional[str] = None,
    strict_files: bool = True,
    return_eid: bool = True,
):
    """
    Split into train/val/test and return three dataset instances.
    If `transform` is provided and `train_transform` is None, use it for train (backward-compatible).
    """
    if train_transform is None and transform is not None:
        train_transform = transform

    full = DXAEidDicomMetaDataset(
        root_dir=root_dir,
        metadata_tsv=metadata_tsv,
        column_map=column_map,
        img_size=img_size,
        window=window,
        transform=None,
        dropna=dropna,
        categorical_maps=categorical_maps,
        eid_column=eid_column,
        strict_files=strict_files,
        return_eid=return_eid,
    )

    from sklearn.model_selection import train_test_split
    random.seed(seed); np.random.seed(seed)

    all_idx = list(range(len(full)))
    train_idx, tmp_idx = train_test_split(all_idx, test_size=test_size, random_state=seed)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=val_size, random_state=seed)

    # Lightweight subset views (share parent; only transform and index differ)
    def subset(indices: List[int], transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]):
        class _Sub(DXAEidDicomMetaDataset):
            def __init__(self, parent: DXAEidDicomMetaDataset, indices_: List[int],
                         transform_: Optional[Callable[[torch.Tensor], torch.Tensor]]):
                self.__dict__ = parent.__dict__.copy()
                self.idx_map = indices_
                self.transform = transform_

            def __len__(self) -> int:
                return len(self.idx_map)

            def __getitem__(self, i: int):
                return super().__getitem__(self.idx_map[i])

        return _Sub(full, indices, transform_fn)

    train_set = subset(train_idx, train_transform)
    val_set   = subset(val_idx,   val_transform)
    test_set  = subset(test_idx,  test_transform)
    return train_set, val_set, test_set


