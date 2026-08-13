import numpy as np
from typing import Dict, Any, List, Tuple, Optional

NOTES = {
    "depth": \
    """Each item lists:
    - p: patch id, starting from 1, which is the upper-left patch
    - c: coordinates of the patch center
    - r: range of depth values for the patch
    - b: dominant object label of the patch. This is optional and may not be present.
    Depth values are between 0 (items in the back) and 1 (items in the front). Coordinates are normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right.
    """,
    
    "correspondence": \
    """Each item lists one keypoint correspondence between two images:
    - p: running id
    - c: point in the first (reference) image
    - r: corresponding point in the second (target) image
    - b: dominant object label of the patch. This is optional and may not be present.
    All coordinates are normalized: (0,0)=top-left, (1000,1000)=bottom-right.""",

    "flow": \
    """Each item lists:
    - p: patch id, starting from 1, which is the upper-left patch
    - c: coordinates of the patch center
    - r: global camera motion in this patch: "left" or "right"
    - b: dominant object label of the patch. This is optional and may not be present.
    Global motion is computed from mean horizontal flow within the patch. Coordinates are normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right.
    """,

    "jigsaw": \
    """Each item lists:
    - p: which border of the hole ('left' or 'top')
    - i: candidate image ID ("A" or "B")
    - c: candidate-piece strip bbox for that border, in piece coordinates, as x0,y0,x1,y1 normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right
    - r: dictionary of average structural, color and edge similarity of that border
    Scores are in [0,1] where higher is better.
    """,

    "similarity": \
    """Each item lists the similarity of a (ref, target) patch pair:
    - p: patch id (row-major), starting at 1 for the upper-left patch
    - c: which target image the pair uses ('A' or 'B')
    - r: similarity score for that patch
    Scores are in [0,1] where higher is better.
    """,

    "semantic": \
    """Each item lists a target candidate and its similarity:
    - p: which target point ('A', 'B', 'C', 'D')
    - c: target point coordinate [x, y] (pixel coordinates as provided normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right)
    - r: similarity score for each point (higher is better)
    """,

    "object-detection": \
    """Each item lists a detected object:
    - p: detection id (running number starting from 1)
    - c: bounding box coordinates [x0, y0, x1, y1]
    - r: detection confidence score (0-1)
    - b: object category label
    Coordinates are normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right.
    """
}

def _row_to_jigsaw_items(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for tag in ("A", "B"):
        cand = row[tag]

        if "left" in cand and "left_bbox" in cand:
            items.append({
                "p": "left",
                "i": tag,
                "c": ",".join(str(int(v)) for v in cand["left_bbox"]),
                "r": {
                    "structural-similarity": float(cand["left"]["structural-similarity"]),
                    "color-similarity":      float(cand["left"]["color-similarity"]),
                    "edge-similarity":       float(cand["left"]["edge-similarity"]),
                }
            })

        if "top" in cand and "top_bbox" in cand:
            items.append({
                "p": "top",
                "i": tag,
                "c": ",".join(str(int(v)) for v in cand["top_bbox"]),
                "r": {
                    "structural-similarity": float(cand["top"]["structural-similarity"]),
                    "color-similarity":      float(cand["top"]["color-similarity"]),
                    "edge-similarity":       float(cand["top"]["edge-similarity"]),
                }
            })
    return items


def _grid_iter(H: int, W: int, rows: int, cols: int):
    patch = 0
    y_edges = np.linspace(0, H, rows+1, dtype=int)
    x_edges = np.linspace(0, W, cols+1, dtype=int)
    
    for r in range(rows):
        for c in range(cols):
            patch += 1  # Patch IDs start at 1
            y0, y1 = y_edges[r], y_edges[r+1]
            x0, x1 = x_edges[c], x_edges[c+1]
            xc = (x0 + x1 - 1) // 2
            yc = (y0 + y1 - 1) // 2
            yield patch, y0, y1, x0, x1, xc, yc

def _dominant_label(seg_patch: np.ndarray) -> Tuple[int, float]:
    if seg_patch.size == 0:
        return -1, 0.0  # Empty patch case
    flat = seg_patch.reshape(-1)
    vals, counts = np.unique(flat, return_counts=True)
    idx = int(np.argmax(counts))
    label = int(vals[idx])
    frac = float(counts[idx] / float(seg_patch.size))
    return label, frac

def _name_for(label_id: int, class_names: Optional[Dict[int,str]]) -> str:
    if label_id < 0:
        return "unknown"
    if class_names and label_id in class_names:
        return str(class_names[label_id])
    return f"class_{label_id}"

def _norm_xy(x, y, W, H):
    return [int(round(x * 1000.0 / W)), int(round(y * 1000.0 / H))]

def _adjacent_pairs(rows: int, cols: int):
    def pid(r, c):
        return r*cols + c + 1
    
    for r in range(rows):
        for c in range(cols):
            p = pid(r,c)
            if c+1 < cols:
                yield (p, pid(r, c+1))
            if r+1 < rows:
                yield (p, pid(r+1, c))

def _round_dict(d: Dict[str,float], ndigits: int = 6) -> Dict[str,float]:
    avgdct = {'similarity': 0}
    dct = {k: (float(round(v, ndigits)) if isinstance(v, (int, float)) else v) for k,v in d.items()}
    avgdct['similarity'] = (dct['structural-similarity'] + dct['color-similarity'] + dct['edge-similarity'])/3.0
    return avgdct

def emit_perception_program(
    modality: str,
    field: Optional[np.ndarray | Dict[str, Any]] = None,
    seg: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    grid: Tuple[int,int] = (10,10),
    add_relations: bool = True,
    tau: float = 0.15,
    relation_cap: int = 1000):

    modality = modality.lower()

    if modality == "jigsaw":
        assert isinstance(field, dict), "For modality='jigsaw', pass a dict field."
        items = _row_to_jigsaw_items(field)
        prog_json: Dict[str, Any] = {
            "modality": "jigsaw",
            "granularity": "border",
            "items": items,
            "relations": [],
            "note": NOTES["jigsaw"],
        }

        header = [
            "<perceptionprogram>",
            "modality: jigsaw",
            "granularity: border",
            f'note: "{NOTES["jigsaw"].strip()}"',
            "items:"
        ]

        item_lines = []
        for it in items:
            r_repr = repr(_round_dict(it["r"], 6))
            item_lines.append(f"  - p: {it['p']}")
            item_lines.append(f"    i: \"{it['i']}\"")
            item_lines.append(f"    c: {it['c']}") # x0,y0,x1,y1
            item_lines.append(f"    r: {r_repr}")

        text_block = "\n".join(header + item_lines + ["</perceptionprogram>"])
        return text_block, prog_json

    elif modality == "semantic":
        assert isinstance(field, dict), "For modality='semantic', pass a row dict with 'scores' and 'tgt_coords'."
        scores = field.get("scores", {})
        tgt_coords = field.get("tgt_coords", [])

        if tgt_coords and isinstance(tgt_coords[0], (int, float)):
            assert len(tgt_coords) == 8, "tgt_coords must be 4 pairs or a flat list of length 8."
            tgt_coords = [[int(tgt_coords[0]), int(tgt_coords[1])],
                        [int(tgt_coords[2]), int(tgt_coords[3])],
                        [int(tgt_coords[4]), int(tgt_coords[5])],
                        [int(tgt_coords[6]), int(tgt_coords[7])]]
        else:
            assert len(tgt_coords) == 4 and all(len(p)==2 for p in tgt_coords), "tgt_coords must be 4 [x,y] pairs."
            tgt_coords = [[int(p[0]), int(p[1])] for p in tgt_coords]

        def _score_for(letter: str) -> float:
            return float(scores.get(f"({letter})",
                        scores.get(letter, 0.0)))

        letters = ["A", "B", "C", "D"]
        items = []
        for i, L in enumerate(letters):
            items.append({
                "p": L,
                "c": [int(tgt_coords[i][0]), int(tgt_coords[i][1])],
                "r": float(_score_for(L)),
            })

        prog_json = {
            "modality": "semantic",
            "granularity": "point",
            "items": items,
            "relations": [],
            "note": NOTES["semantic"],
        }

        header = [
            "<perceptionprogram>",
            "modality: semantic",
            "granularity: point",
            f'note: "{NOTES["semantic"].strip()}"',
            "items:",
        ]
        item_lines = []
        for it in items:
            item_lines.append(f'  - p: "{it["p"]}"')
            item_lines.append(f"    c: [{it['c'][0]}, {it['c'][1]}]")
            item_lines.append(f"    r: {it['r']:.6f}")
        text_block = "\n".join(header + item_lines + ["</perceptionprogram>"])
        return text_block, prog_json

    elif modality == "similarity":
        assert isinstance(field, dict), "For modality='similarity', pass field={'A': np.ndarray, 'B': np.ndarray} (arrays optional per key)."
        maps = {}
        for k in ("A", "B"):
            if k in field and field[k] is not None:
                arr = np.asarray(field[k])
                assert arr.ndim == 2, f"Similarity map for {k} must be 2D, got shape {arr.shape}."
                maps[k] = arr

        assert maps, "No similarity maps found in field. Provide at least one of {'A','B'}."
        shapes = {v.shape for v in maps.values()}
        assert len(shapes) == 1, f"Similarity maps must share shape, got: {shapes}"
        rows, cols = next(iter(shapes))
        items = []
        p_counter = 0
        for tgt_label, sim2d in maps.items():
            for r_idx in range(rows):
                for c_idx in range(cols):
                    p_counter += 1
                    items.append({
                        "p": int(p_counter),
                        "c": tgt_label, # 'A' or 'B'
                        "r": float(sim2d[r_idx, c_idx])
                    })

        prog_json = {
            "modality": "similarity",
            "granularity": "patch",
            "grid": {"rows": int(rows), "cols": int(cols)},
            "items": items,
            "relations": [],
            "note": NOTES["similarity"],
        }
        
        header = [
            "<perceptionprogram>",
            "modality: similarity",
            "granularity: patch",
            "grid:",
            f"  rows: {rows}",
            f"  cols: {cols}",
            f'note: "{NOTES["similarity"].strip()}"',
            "items:",
        ]
        item_lines = []
        for it in items:
            item_lines.append(f"  - p: {it['p']}")
            item_lines.append(f"    c: \"{it['c']}\"")
            item_lines.append(f"    r: {it['r']:.6f}")
        text_block = "\n".join(header + item_lines + ["</perceptionprogram>"])

        return text_block, prog_json
    
    elif modality == "correspondence":
        assert isinstance(field, dict), \
        "For modality='correspondence', pass a dict: {'kpts0','kpts1','W1','H1','W2','H2', ('conf' optional)}"
        kpts0 = np.asarray(field["kpts0"], dtype=np.float32)  # Nx2 (x,y) in image_1
        kpts1 = np.asarray(field["kpts1"], dtype=np.float32)  # Nx2 (x,y) in image_2
        W1, H1 = int(field["W1"]), int(field["H1"])
        W2, H2 = int(field["W2"]), int(field["H2"])
        assert kpts0.shape == kpts1.shape and kpts0.shape[1] == 2, "kpts0/kpts1 must be Nx2"
        
        items: List[Dict[str, Any]] = []
        for i in range(len(kpts0)):
            x0, y0 = float(kpts0[i,0]), float(kpts0[i,1])
            x1, y1 = float(kpts1[i,0]), float(kpts1[i,1])
            c_ref = _norm_xy(x0, y0, W1, H1)
            r_tgt = _norm_xy(x1, y1, W2, H2)
            items.append({"p": int(i+1), "c": c_ref, "r": r_tgt})
            prog_json: Dict[str, Any] = {
                "modality": "correspondence",
                "granularity": "keypoint",
                "items": items,
                "relations": [], 
                "note": NOTES["correspondence"],
            }
        header = [
            "<perceptionprogram>",
            "modality: correspondence",
            "granularity: keypoint",
            f'note: "{NOTES["correspondence"].strip()}"',
            "items:"
        ]
        item_lines = []
        for it in items:
            item_lines.append(f"  - p: {it['p']}")
            item_lines.append(f"    c: [{it['c'][0]}, {it['c'][1]}]")
            item_lines.append(f"    r: [{it['r'][0]}, {it['r'][1]}]")
        footer = ["</perceptionprogram>"]
        text_block = "\n".join(header + item_lines + footer)
        return text_block, prog_json
    
    elif modality == "object-detection":
        assert isinstance(field, list), "For modality='object-detection', pass a list of detection dicts"
        
        items = []
        for i, det in enumerate(field, start=1):
            assert "label" in det and "confidence" in det and "bbox" in det, \
                "Each detection must have 'label', 'confidence', and 'bbox' fields"
            bbox = det["bbox"]
            assert len(bbox) == 4, "bbox must be [x0, y0, x1, y1]"
            items.append({
                "p": int(i),
                "c": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "r": float(det["confidence"]),
                "b": str(det["label"])
            })
        
        prog_json: Dict[str, Any] = {
            "modality": "object-detection",
            "granularity": "bounding-box",
            "items": items,
            "relations": [],
            "note": NOTES["object-detection"],
        }
        
        header = [
            "<perceptionprogram>",
            "modality: object-detection",
            "granularity: bounding-box",
            f'note: "{NOTES["object-detection"].strip()}"',
            "items:"
        ]
        item_lines = []
        for it in items:
            bbox_str = f"[{it['c'][0]}, {it['c'][1]}, {it['c'][2]}, {it['c'][3]}]"
            item_lines.append(f"  - p: {it['p']}")
            item_lines.append(f"    c: {bbox_str}")
            item_lines.append(f"    r: {it['r']:.4f}")
            item_lines.append(f"    b: \"{it['b']}\"")
        
        footer = ["</perceptionprogram>"]
        text_block = "\n".join(header + item_lines + footer)
        return text_block, prog_json
    
    else:
        if modality == "segmentation":
            H, W = seg.shape
        elif modality == "flow":
            H, W, _ = field.shape
        elif modality == "jigsaw":
            assert isinstance(field, dict)
        else:
            H, W = field.shape
        rows, cols = grid
        items: List[Dict[str, Any]] = []
        means = {} 
        labels_dom = {}
        for p, y0,y1,x0,x1, xc,yc in _grid_iter(H, W, rows, cols):
            xc_norm = int(xc * 1000 / W)
            yc_norm = int(yc * 1000 / H)
            it = {"p": int(p), "c": [xc_norm, yc_norm]}  
            if modality == "segmentation":
                lid, frac = _dominant_label(seg[y0:y1, x0:x1])
                it["r"] = float(frac) # r = coverage fraction
                it["b"] = _name_for(lid, class_names) # b = class name
                means[p] = frac # Store for relation logic
                labels_dom[p] = lid
            else:
                if modality == "flow":
                    assert field.ndim == 3 and field.shape[2] == 2, "flow must be [H,W,2]"
                    patch = field[y0:y1, x0:x1, 0] # horizontal flow
                    if patch.size == 0:
                        u_val = 0.0
                    else:
                        u_val = np.mean(patch)
                    # global camera motion direction in the patch
                    motion = "left" if u_val < 0.0 else "right"
                    it["r"] = motion
                    m = u_val
                else:
                    # For depth: compute min, max and mean of the patch
                    vals = field[y0:y1, x0:x1].astype(np.float32).reshape(-1)
                    if vals.size == 0:
                        rmin = rmax = float("nan")
                        m = float("nan")
                        it["r"] = [rmin, rmax] # r = [min, max] range
                    else:
                        rmin = float(np.min(vals))
                        rmax = float(np.max(vals))
                        m = float(np.mean(vals))
                        it["r"] = [rmin, rmax]
                
                if seg is not None:
                    lid, _ = _dominant_label(seg[y0:y1, x0:x1])
                    it["b"] = _name_for(lid, class_names)   # b = class name
                    labels_dom[p] = lid
                means[p] = m # Store statistic for relations
            items.append(it)

        relations: List[List[Any]] = []
        if add_relations:
            if modality == "depth":
                for a,b in _adjacent_pairs(rows, cols):
                    ma, mb = means.get(a, np.nan), means.get(b, np.nan)
                    if np.isnan(ma) or np.isnan(mb):
                        continue
                    
                    # If patch 'a' has significantly larger depth value, it's in front
                    if ma > mb + tau:
                        relations.append([f"p={a}", "in_front_of", f"p={b}"])
                    elif mb > ma + tau:
                        relations.append([f"p={b}", "in_front_of", f"p={a}"])
                    if len(relations) >= relation_cap: 
                        break

        modality_notes = NOTES.get(modality, "")
        prog_json: Dict[str, Any] = {
            "modality": modality,
            "granularity": "patch",
            "grid": {"rows": rows, "cols": cols},
            "items": items,
            "relations": relations,
            "note": modality_notes,
        }

        header = [
            "<perceptionprogram>",
            f"modality: {modality}",
            "granularity: patch",
            "grid:",
            f"  rows: {rows}",
            f"  cols: {cols}",
            f'note: "{modality_notes.strip()}"',
            "items:"
        ]

        item_lines = []
        for it in items:
            item_lines.append(f"  - p: {it['p']}")
            item_lines.append(f"    c: [{it['c'][0]}, {it['c'][1]}]")
            # r is either [min, max] for depth/albedo or a single fraction for segmentation
            if isinstance(it["r"], list):
                item_lines.append(f"    r: [{it['r'][0]:.4g}, {it['r'][1]:.4g}]")
            else:
                if isinstance(it["r"], str):
                    item_lines.append(f"    r: {it['r']}")
                else:
                    item_lines.append(f"    r: {it['r']:.4g}")
            if "b" in it:
                item_lines.append(f"    b: \"{it['b']}\"")

        rel_lines = []
        if relations:
            rel_lines.append("relations:")
            for rel in relations[:relation_cap]:
                a, op, b = rel
                rel_lines.append(f"  - [{a}, {op}, {b}]")

        footer = ["</perceptionprogram>"]
        text_block = "\n".join(header + item_lines + rel_lines + footer)
        
        return text_block, prog_json