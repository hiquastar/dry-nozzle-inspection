"""
create_splits_reverse.py  —  Build data/splits_reverse/ from per-source COCO files.

The repo stores annotations as one COCO JSON per source in
  data/annotations/v2/
Splits are virtual — defined here by source-video tag (frame-range-aware
for the chronological 20250509 video).

Reverse split layout
  Train : 20250509 frames 0-213, 20250508, CAM2-V1, CAM2-V2, CAM2-V3
  Val   : 20250509 frames 230-261, CAM1-V1
  Test  : 20250509 frames 275-304, 20250510, CAM1-V2, CAM1-V3

Run from repo root:
    python data/scripts/create_splits_reverse.py
"""

import json, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[2]  # scripts -> data -> repo root
DATA_ROOT  = REPO_ROOT / 'data' / 'annotations' / 'v2'
OUT_DIR    = REPO_ROOT / 'data' / 'splits_reverse'

SOURCE_FILES = {
    '20250508':  '20250508_annotations.coco.json',
    '20250509':  '20250509_annotations.coco.json',
    '20250510':  '20250510_annotations.coco.json',
    'CAM1 - V1': 'CAM1 - V1_annotations.coco.json',
    'CAM1 - V2': 'CAM1 - V2_annotations.coco.json',
    'CAM1 - V3': 'CAM1 - V3_annotations.coco.json',
    'CAM2 - V1': 'CAM2 - V1_annotations.coco.json',
    'CAM2 - V2': 'CAM2 - V2_annotations.coco.json',
    'CAM2 - V3': 'CAM2 - V3_annotations.coco.json',
}

# 20250509 sub-ranges. Frames in gaps (214-229, 262-274) are dropped.
RANGES_20250509 = [
    ('20250509_frames_0-213',   range(0,   214)),
    ('20250509_frames_230-261', range(230, 262)),
    ('20250509_frames_275-304', range(275, 305)),
]

# Reverse split: CAM2 in train, CAM1 split into val/test.
ASSIGNMENT = {
    'train': {
        '20250509_frames_0-213',
        '20250508',
        'CAM2 - V1',
        'CAM2 - V2',
        'CAM2 - V3',
    },
    'val': {
        '20250509_frames_230-261',
        'CAM1 - V1',
    },
    'test': {
        '20250509_frames_275-304',
        '20250510',
        'CAM1 - V2',
        'CAM1 - V3',
    },
}
SRC_TO_SPLIT = {src: spl for spl, srcs in ASSIGNMENT.items() for src in srcs}


def tag_20250509(file_name: str):
    """Return ('20250509_frames_a-b', frame_int) or (None, None) if excluded."""
    stem = Path(file_name).stem  # '20250509_000213'
    parts = stem.split('_')
    try:
        frame = int(parts[-1])
    except (ValueError, IndexError):
        return None, None
    for tag, rng in RANGES_20250509:
        if frame in rng:
            return tag, frame
    return None, frame  # excluded gap


def load_pooled():
    """Read every per-source COCO file, tag images, re-id to avoid collisions."""
    all_images, all_annotations, categories = [], [], None
    img_id_counter = 1
    ann_id_counter = 1
    excluded_20250509 = 0

    for source_key, fname in SOURCE_FILES.items():
        path = DATA_ROOT / fname
        if not path.exists():
            print(f'WARN: missing {path}')
            continue
        with open(path) as f:
            d = json.load(f)
        if categories is None:
            categories = d.get('categories', [])

        # Old image_id -> new image_id (only for kept images)
        id_remap = {}

        for img in d['images']:
            if source_key == '20250509':
                tag, frame = tag_20250509(img['file_name'])
                if tag is None:
                    excluded_20250509 += 1
                    continue
            else:
                tag = source_key

            old_id = img['id']
            new_id = img_id_counter
            img_id_counter += 1
            id_remap[old_id] = new_id

            new_img = dict(img)
            new_img['id'] = new_id
            new_img['source_video'] = tag
            all_images.append(new_img)

        for ann in d['annotations']:
            if ann['image_id'] not in id_remap:
                continue
            new_ann = dict(ann)
            new_ann['image_id'] = id_remap[ann['image_id']]
            new_ann['id'] = ann_id_counter
            ann_id_counter += 1
            all_annotations.append(new_ann)

    if excluded_20250509:
        print(f'  excluded {excluded_20250509} 20250509 frames in gap ranges')
    return all_images, all_annotations, categories


def build_splits(all_images, all_annotations, categories):
    ann_by_img = defaultdict(list)
    for ann in all_annotations:
        ann_by_img[ann['image_id']].append(ann)

    splits = {s: {'images': [], 'annotations': [], 'categories': categories}
              for s in ('train', 'val', 'test')}
    unassigned = []

    for img in all_images:
        src = img.get('source_video', '')
        split = SRC_TO_SPLIT.get(src)
        if split is None:
            unassigned.append(src)
            continue
        splits[split]['images'].append(img)
        splits[split]['annotations'].extend(ann_by_img[img['id']])

    if unassigned:
        print(f'WARN: {len(unassigned)} images had unrecognised source_video tag '
              f'(samples: {list(set(unassigned))[:5]})')

    return splits


def summarise(splits):
    print()
    print('%-7s  %7s  %6s  %7s  %6s  %6s  %9s' %
          ('Split', 'Frames', 'spray', 'facing', 'no_sp', 'big', 'Total'))
    print('-' * 60)
    for name, d in splits.items():
        cls = Counter()
        for ann in d['annotations']:
            cls[ann.get('_meta', {}).get('original_class', 'unknown')] += 1
        sp = cls.get('nozzle_spray', 0)
        fa = cls.get('nozzle_facing', 0)
        ns = cls.get('nozzle_no_spray', 0)
        bi = cls.get('big_nozzle', 0)
        tot = sp + fa + ns + bi
        print('%-7s  %7d  %6d  %7d  %6d  %6d  %9d' %
              (name, len(d['images']), sp, fa, ns, bi, tot))
    print()
    print('Camera positions per split:')
    for name, d in splits.items():
        srcs = sorted({i.get('source_video', '?') for i in d['images']})
        print(f'  {name}: {", ".join(srcs)}')


def verify_no_overlap(splits):
    per_split = {s: {i['id'] for i in d['images']} for s, d in splits.items()}
    names = list(per_split)
    ok = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = per_split[names[i]] & per_split[names[j]]
            if overlap:
                print(f'ERROR: {len(overlap)} image_ids in both {names[i]} and {names[j]}')
                ok = False
    if ok:
        print('\nVerification PASSED: no image_id appears in more than one split.')


def main():
    if not DATA_ROOT.exists():
        sys.exit(f'ERROR: data root not found: {DATA_ROOT}')

    print(f'Reading per-source COCO files from: {DATA_ROOT}')
    all_images, all_annotations, categories = load_pooled()
    print(f'  pooled: {len(all_images)} images, {len(all_annotations)} annotations')

    splits = build_splits(all_images, all_annotations, categories)
    summarise(splits)
    verify_no_overlap(splits)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in splits.items():
        out = OUT_DIR / f'{name}.coco.json'
        with open(out, 'w') as f:
            json.dump(d, f)
        print(f'Wrote {out.name}  ({len(d["images"])} images, {len(d["annotations"])} anns)')

    print(f'\nDone. Files in: {OUT_DIR}')


if __name__ == '__main__':
    main()
