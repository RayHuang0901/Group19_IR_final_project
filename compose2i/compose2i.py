import os
import sys
import json
import random
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, Counter

from datasets import load_dataset
from openai import OpenAI

import torch
import open_clip
import numpy as np
from tqdm import tqdm
import faiss
from PIL import Image

import time
from transformers import BlipProcessor, BlipForConditionalGeneration


# ============================
# 全域 LLM client（避免每次重建）
# ============================
# 需先設好環境變數: OPENAI_API_KEY

LLM_CLIENT = OpenAI()
LLM_MODEL = "gpt-4o-mini"

BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# =========================
# Config
# =========================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TOTAL_IMAGES = 5000
MIN_PER_TRIPLE = 15
MIN_COLORS_PER_PAIR = 2

KS = (1, 5, 10)
MAX_K = 50


NUM_Q_BASELINE = 300
NUM_Q_OURS = 300

# baseline fusion alpha
FUSION_ALPHA = 0.5


# ============================
# LLM 相關工具函式
# ============================

def _length_instruction(length_mode: Optional[str]) -> str:
    """根據長度模式回傳一句放進 prompt 的說明文字。"""
    if length_mode is None:
        return ""
    if length_mode == "short":
        return "Limit the output to roughly 10-20 English words."
    if length_mode == "medium":
        return "Limit the output to roughly 40-50 English words."
    if length_mode == "long":
        return "You may use up to roughly 100 English words."
    return ""

def llm_compose_caption(
    ref_blip_caption: str,
    user_mod_text: str,
    mode: str = "detail",
    length_mode: Optional[str] = None,
) -> Optional[str]:
    length_instr = _length_instruction(length_mode)

    if mode == "detail":
        system_part = f"""
You are helping build a fashion image retrieval system.

Given:
- A visual caption of a reference clothing item.
- A user modification request (e.g., change color).

Write ONE English sentence describing the TARGET clothing item AFTER applying the modification.

Rules:
- Focus strictly on clothing appearance: color, fit, cut, length, pattern, material, key design details.
- Do NOT mention people, background, scene, brand, price.
- Do NOT invent attributes not implied by the reference caption + modification.
{length_instr}
""".strip()

    elif mode == "keywords":
        system_part = f"""
You are helping build a fashion image retrieval system.

Given:
- A visual caption of a reference clothing item.
- A user modification request.

Return a comma-separated list of concise visual keywords describing the TARGET clothing item AFTER applying the modification.

Rules:
- Only visual clothing attributes: color, fit, cut, length, pattern, material, key details.
- No people/background/scene/brand/price.
- Return ONLY comma-separated keywords, no extra text.
{length_instr}
""".strip()

    elif mode == "conversational":
        system_part = f"""
You are helping build a fashion image search system.

Given:
- A visual caption of a reference clothing item.
- A user modification request.

Write ONE conversational search query a user might type to find the TARGET clothing item AFTER applying the modification.

Rules:
- Focus on clothing appearance only.
- No people/background/scene/brand/price.
- Do NOT invent attributes.
{length_instr}
""".strip()

    else:
        system_part = f"""
You are helping build a fashion image retrieval system.
Write ONE English sentence describing the TARGET clothing item AFTER applying the modification.
{length_instr}
""".strip()

    prompt = f"""
{system_part}

Reference image caption:
\"\"\"{ref_blip_caption}\"\"\"

User modification:
\"\"\"{user_mod_text}\"\"\"

Output:
""".strip()

    last_err = None
    for attempt in range(1, 6):
        try:
            resp = LLM_CLIENT.responses.create(
                model=LLM_MODEL,
                input=prompt,
                max_output_tokens=120,
                temperature=0.4,
                timeout=15
            )
            text = resp.output[0].content[0].text.strip()
            if text:
                return text
            raise RuntimeError("Empty LLM response")

        except Exception as e:
            last_err = e
            print(f"[LLM WARNING] attempt {attempt}/5 failed: {e}")
            if attempt < 5:
                sleep_time = 2 ** (attempt - 1)
                print(f"[LLM] retrying after {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    print(f"[LLM ERROR] failed after retries: {last_err}")
    return None

# ============================
# 建立資料集
# ============================

def build_small_ds_min15_per_triple(
    ds,
    total=5000,
    min_per_triple=15,
    min_colors_per_pair=2,
    seed=42,
):
    rng = random.Random(seed)

    # 1) 建 triple -> indices
    triple_to_indices = defaultdict(list)
    pair_to_color_counts = defaultdict(lambda: defaultdict(int))

    for idx, s in enumerate(ds):
        c1 = s.get("category1", None)
        c2 = s.get("category2", None)
        col = s.get("color", None)
        if c1 is None or c2 is None or col is None:
            continue
        triple = (c1, c2, col)
        triple_to_indices[triple].append(idx)
        pair_to_color_counts[(c1, c2)][col] += 1

    # 2) eligible triples: 每個 (c1,c2,col) 至少 min_per_triple
    eligible_triples = {
        triple: idxs for triple, idxs in triple_to_indices.items()
        if len(idxs) >= min_per_triple
    }

    # 3) valid pairs: 至少 min_colors_per_pair 個 eligible colors
    pair_to_colors = defaultdict(list)  # (c1,c2) -> list of (color, count, idxs)
    for (c1, c2, col), idxs in eligible_triples.items():
        pair_to_colors[(c1, c2)].append((col, len(idxs), idxs))

    valid_pairs = {
        pair: sorted(colors, key=lambda x: x[1], reverse=True)  # 依 count 由大到小
        for pair, colors in pair_to_colors.items()
        if len(colors) >= min_colors_per_pair
    }

    print(f"[Info] Eligible triples (>= {min_per_triple}): {len(eligible_triples)}")
    print(f"[Info] Valid pairs (>= {min_colors_per_pair} colors): {len(valid_pairs)}")

    if len(valid_pairs) == 0:
        print("[ERROR] No valid (category1,category2) pairs satisfy the constraint.")
        sys.exit(1)

    # 4) 你最多能選幾個 triple？（每個 triple base 需要 min_per_triple 張）
    triple_budget = total // min_per_triple
    if triple_budget < 2:
        print("[ERROR] total too small vs min_per_triple; cannot even pick 2 triples.")
        sys.exit(1)

    # 5) 先以 pair 為單位：每個新 pair 至少選 2 個 color triples（保證兩色）
    #    用 greedy：優先選「總量大」的 pair，確保後續補到 total 不會缺圖
    pair_sorted = sorted(
        valid_pairs.items(),
        key=lambda kv: sum(c[1] for c in kv[1]),  # pair 內 eligible 圖片總數
        reverse=True
    )

    selected_triples = []  # list of (c1,c2,col)
    selected_triple_to_idxs = {}  # triple -> idxs

    # 5a) 先選一些 pairs，每個 pair 先拿 top2 colors
    for (pair, colors) in pair_sorted:
        if len(selected_triples) + min_colors_per_pair > triple_budget:
            break
        # 取前 min_colors_per_pair 個顏色（count 最大的）
        for j in range(min_colors_per_pair):
            col, cnt, idxs = colors[j]
            triple = (pair[0], pair[1], col)
            selected_triples.append(triple)
            selected_triple_to_idxs[triple] = idxs

    if len(selected_triples) < 2:
        print("[ERROR] Could not select enough triples under budget to satisfy pair>=2 colors.")
        sys.exit(1)

    # 5b) 如果 triple_budget 還有空間：再加「額外顏色」進已選 pairs（同 pair 的其他 colors）
    #     也是 greedy：先把剩下可選的 triples 丟進 candidate，依 count 由大到小補
    if len(selected_triples) < triple_budget:
        candidates = []
        selected_pairs = set((t[0], t[1]) for t in selected_triples)

        for (pair, colors) in valid_pairs.items():
            if pair not in selected_pairs:
                continue
            # 這個 pair 已經選過兩個色了，剩下的也能選（如果 budget 允許）
            for (col, cnt, idxs) in colors[min_colors_per_pair:]:
                triple = (pair[0], pair[1], col)
                candidates.append((cnt, triple, idxs))

        candidates.sort(reverse=True, key=lambda x: x[0])

        for cnt, triple, idxs in candidates:
            if len(selected_triples) >= triple_budget:
                break
            if triple in selected_triple_to_idxs:
                continue
            selected_triples.append(triple)
            selected_triple_to_idxs[triple] = idxs

    # 6) 檢查選到的 triples 是否足夠湊滿 total
    total_available = sum(len(selected_triple_to_idxs[t]) for t in selected_triples)
    base_needed = len(selected_triples) * min_per_triple

    print(f"[Info] Selected triples: {len(selected_triples)} (budget={triple_budget})")
    print(f"[Info] Base needed: {base_needed}, total available in selected triples: {total_available}")

    if base_needed > total:
        print("[ERROR] Base samples already exceed total. Reduce min_per_triple or #selected triples.")
        sys.exit(1)

    if total_available < total:
        print(f"[ERROR] Not enough images in selected triples to fill total={total}.")
        print(f"        available={total_available}. Try lowering constraints or using larger total.")
        sys.exit(1)

    # 7) 保底抽樣：每個 triple 抽 min_per_triple 張
    selected = []
    selected_set = set()

    leftovers = []  # 之後用來補到 total

    for triple in selected_triples:
        idxs = selected_triple_to_idxs[triple]
        idxs = idxs.copy()
        rng.shuffle(idxs)

        base = idxs[:min_per_triple]
        rest = idxs[min_per_triple:]

        for i in base:
            if i not in selected_set:
                selected.append(i)
                selected_set.add(i)

        for i in rest:
            if i not in selected_set:
                leftovers.append(i)

    # 8) 補到 total
    need = total - len(selected)
    if len(leftovers) < need:
        print(f"[ERROR] Not enough leftovers to fill total={total}. need={need}, leftovers={len(leftovers)}")
        sys.exit(1)

    rng.shuffle(leftovers)
    selected.extend(leftovers[:need])

    # 9) 最後 sanity check：small_ds 內每個 triple 都 >= min_per_triple
    triple_cnt = defaultdict(int)
    pair_colors = defaultdict(set)

    # 用 selected indices 重新計算
    for idx in selected:
        s = ds[int(idx)]
        c1, c2, col = s["category1"], s["category2"], s["color"]
        triple_cnt[(c1, c2, col)] += 1
        pair_colors[(c1, c2)].add(col)

    bad_triples = [t for t, cnt in triple_cnt.items() if cnt < min_per_triple]
    bad_pairs = [p for p, cols in pair_colors.items() if len(cols) < min_colors_per_pair]

    if bad_triples:
        print("[ERROR] Some triples violate min_per_triple:", bad_triples[:5])
        sys.exit(1)

    if bad_pairs:
        print("[ERROR] Some pairs violate min_colors_per_pair:", bad_pairs[:5])
        sys.exit(1)

    print("[OK] Sampling satisfied: each (cat1,cat2,color) >= min_per_triple and each (cat1,cat2) has >= min_colors_per_pair colors.")
    return ds.select(selected)


# =========================
# Step 2: BLIP captions (ref image only) + cache
# =========================

def build_blip_captions(small_ds, device, use_cache: bool = True):
    

    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    blip = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)
    blip.eval()

    caps = []
    for i in tqdm(range(len(small_ds)), desc="BLIP captioning"):
        img = small_ds[i]["image"].convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip.generate(**inputs, max_new_tokens=30)
        cap = processor.decode(out[0], skip_special_tokens=True).strip()
        caps.append(cap)


    return caps


# =========================
# Baseline-2
# =========================

def baseline2_compose_query_text_concat_raw(blip_caption: str, user_mod_text: str) -> str:
    cap = blip_caption.strip().strip(".")
    mod = user_mod_text.strip().strip(".")
    return f"{cap}. {mod}."

# =========================
# 建立測試的題目
# =========================

def build_color_queries_strict(
    small_ds,
    seed: int = 42,
    min_pos: int = 15,
    max_queries_per_pair: int = 400,
) -> List[dict]:
    rng = random.Random(seed)

    pair_color_to_indices = defaultdict(lambda: defaultdict(list))
    for idx, s in enumerate(small_ds):
        c1 = s.get("category1", None)
        c2 = s.get("category2", None)
        col = s.get("color", None)
        if c1 is None or c2 is None or col is None:
            continue
        pair_color_to_indices[(c1, c2)][col].append(idx)

    queries = []

    for (cat1, cat2), color_map in pair_color_to_indices.items():
        colors_ok = [c for c, idxs in color_map.items() if len(idxs) >= min_pos]
        if len(colors_ok) < 2:
            continue

        pair_queries = []

        for ref_color, ref_indices in color_map.items():
            if len(ref_indices) == 0:
                continue
            target_colors = [c for c in colors_ok if c != ref_color]
            if not target_colors:
                continue

            for ref_idx in ref_indices:
                tgt_color = rng.choice(target_colors)
                gt_indices = color_map[tgt_color]
                if len(gt_indices) < min_pos:
                    continue

                pair_queries.append({
                    "ref_idx": ref_idx,
                    "user_mod_text": f"change the color to {tgt_color}",
                    "target_color": tgt_color,
                    "target_cat1": cat1,
                    "target_cat2": cat2,
                    "gt_indices": gt_indices,
                    "num_pos": len(gt_indices),
                })

        if len(pair_queries) > max_queries_per_pair:
            rng.shuffle(pair_queries)
            pair_queries = pair_queries[:max_queries_per_pair]

        queries.extend(pair_queries)

    print(f"[Info] Strict composed queries: {len(queries)}")
    if queries:
        pos_counts = sorted(q["num_pos"] for q in queries)
        print(f"[Info] Positives per query: min={pos_counts[0]}, median={pos_counts[len(pos_counts)//2]}, max={pos_counts[-1]}")
    return queries


# =========================
# 評測
# =========================

def compute_metrics_from_relevance(all_relevance: List[List[bool]], ks: Tuple[int, ...] = (1, 5, 10)) -> Dict:
    num_queries = len(all_relevance)
    if num_queries == 0:
        return {"recall": {k: 0.0 for k in ks}, "precision": {k: 0.0 for k in ks}, "MRR": 0.0, "mAP": 0.0}

    ks = tuple(sorted(ks))
    recall_sum = {k: 0.0 for k in ks}
    precision_sum = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    map_sum = 0.0

    for rel in all_relevance:
        num_rel = sum(rel)

        for k in ks:
            topk = rel[:k]
            hit = sum(topk)
            recall_sum[k] += 1.0 if hit > 0 else 0.0
            precision_sum[k] += hit / float(k)

        if num_rel > 0:
            for rank, r in enumerate(rel):
                if r:
                    mrr_sum += 1.0 / float(rank + 1)
                    break

            hit_cnt = 0
            ap = 0.0
            for rank, r in enumerate(rel):
                if r:
                    hit_cnt += 1
                    ap += hit_cnt / float(rank + 1)
            ap /= float(num_rel)
            map_sum += ap

    return {
        "recall": {k: recall_sum[k] / num_queries for k in ks},
        "precision": {k: precision_sum[k] / num_queries for k in ks},
        "MRR": mrr_sum / num_queries,
        "mAP": map_sum / num_queries,
    }


def print_metrics(title: str, metrics: Dict, ks=(1, 5, 10), map_k=50):
    print(f"\n=== {title} ===")
    for k in ks:
        print(f"Recall@{k}:    {metrics['recall'][k]:.4f}")
        print(f"Precision@{k}: {metrics['precision'][k]:.4f}")
    print(f"MRR:            {metrics['MRR']:.4f}")
    print(f"mAP@{map_k}:         {metrics['mAP']:.4f}")


# =========================
# Baseline-1: SigLIP Fusion (no LLM, no BLIP)
# =========================

def eval_baseline_image_text_fusion(
    queries,
    model,
    tokenizer,
    index,
    device,
    img_embeds_np: np.ndarray,
    alpha: float = 0.5,
    num_queries: int = 300,
    ks=(1, 5, 10),
    max_k=50,
):
    rng = random.Random(RANDOM_SEED)
    if len(queries) == 0:
        return compute_metrics_from_relevance([], ks=ks)

    chosen = queries.copy()
    rng.shuffle(chosen)
    chosen = chosen[:min(num_queries, len(chosen))]

    all_relevance = []

    for q in tqdm(chosen, desc=f"Baseline-1 SigLIP Fusion (alpha={alpha})"):
        ref_idx = q["ref_idx"]
        mod_text = q["user_mod_text"]

        v_img = torch.from_numpy(img_embeds_np[ref_idx:ref_idx+1]).to(device)

        with torch.no_grad():
            tokens = tokenizer([mod_text]).to(device)
            v_text = model.encode_text(tokens)
        v_text = v_text / v_text.norm(dim=-1, keepdim=True)

        q_embed = alpha * v_img + (1.0 - alpha) * v_text
        q_embed = q_embed / q_embed.norm(dim=-1, keepdim=True)
        q_np = q_embed.cpu().numpy().astype("float32")

        _, I = index.search(q_np, max_k)
        retrieved = I[0]

        gt_set = set(q["gt_indices"])
        relevance = [(int(idx) in gt_set) for idx in retrieved]
        all_relevance.append(relevance)

    return compute_metrics_from_relevance(all_relevance, ks=ks)


# =========================
# Baseline-2: BLIP caption + concat_raw (no LLM)
# =========================

def eval_baseline_blip_concat_raw(
    blip_caps: List[str],
    queries,
    model,
    tokenizer,
    index,
    device,
    num_queries: int = 300,
    ks=(1, 5, 10),
    max_k=50,
):
    rng = random.Random(RANDOM_SEED)
    if len(queries) == 0:
        return compute_metrics_from_relevance([], ks=ks)

    chosen = queries.copy()
    rng.shuffle(chosen)
    chosen = chosen[:min(num_queries, len(chosen))]

    all_relevance = []

    for q in tqdm(chosen, desc="Baseline-2 BLIP+concat_raw (no LLM)"):
        ref_idx = q["ref_idx"]
        mod_text = q["user_mod_text"]

        ref_cap = blip_caps[ref_idx]
        query_text = baseline2_compose_query_text_concat_raw(ref_cap, mod_text)

        with torch.no_grad():
            tokens = tokenizer([query_text]).to(device)
            feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        q_np = feat.cpu().numpy().astype("float32")

        _, I = index.search(q_np, max_k)
        retrieved = I[0]

        gt_set = set(q["gt_indices"])
        relevance = [(int(idx) in gt_set) for idx in retrieved]
        all_relevance.append(relevance)

    return compute_metrics_from_relevance(all_relevance, ks=ks)


# =========================
# BLIP + LLM strategies
# =========================

def eval_blip_llm(
    blip_caps: List[str],
    queries,
    model,
    tokenizer,
    index,
    device,
    llm_mode: str,
    length_mode: Optional[str],
    num_queries: int = 200,
    ks=(1, 5, 10),
    max_k=50,
):
    rng = random.Random(RANDOM_SEED)
    if len(queries) == 0:
        return compute_metrics_from_relevance([], ks=ks)

    chosen = queries.copy()
    rng.shuffle(chosen)
    chosen = chosen[:min(num_queries, len(chosen))]

    all_relevance = []

    for q in tqdm(chosen, desc=f"Ours BLIP+LLM (mode={llm_mode}, len={length_mode})"):
        ref_idx = q["ref_idx"]
        mod_text = q["user_mod_text"]

        ref_cap = blip_caps[ref_idx]
        target_caption = llm_compose_caption(
            ref_blip_caption=ref_cap,
            user_mod_text=mod_text,
            mode=llm_mode,
            length_mode=length_mode,
        )

        if not target_caption:
            all_relevance.append([False] * max_k)
            continue

        with torch.no_grad():
            tokens = tokenizer([target_caption]).to(device)
            feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        q_np = feat.cpu().numpy().astype("float32")

        _, I = index.search(q_np, max_k)
        retrieved = I[0]

        gt_set = set(q["gt_indices"])
        relevance = [(int(idx) in gt_set) for idx in retrieved]
        all_relevance.append(relevance)

    return compute_metrics_from_relevance(all_relevance, ks=ks)


# =========================
# Main
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device =", device)

    
    ds = load_dataset("Marqo/deepfashion-inshop", split="data")

    small_ds = build_small_ds_min15_per_triple(
        ds,
        total=TOTAL_IMAGES,
        min_per_triple=MIN_PER_TRIPLE,
        min_colors_per_pair=MIN_COLORS_PER_PAIR,
        seed=RANDOM_SEED,
    )
    print("[Info] small_ds size:", len(small_ds))

    # Load SigLIP
    print("Loading SigLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
    model.to(device)
    model.eval()

    # 建立 FAISS index
    print("Computing image embeddings...")
    img_feats = []
    for s in tqdm(small_ds, desc="Encoding images"):
        img = s["image"].convert("RGB")
        img_t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encode_image(img_t)
        f = f / f.norm(dim=-1, keepdim=True)
        img_feats.append(f.cpu().numpy())
    img_feats = np.vstack(img_feats).astype("float32")

    dim = img_feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(img_feats)

    # 建立評測題目
    queries = build_color_queries_strict(
        small_ds,
        seed=RANDOM_SEED,
        min_pos=MIN_PER_TRIPLE,
        max_queries_per_pair=400,
    )

    if len(queries) == 0:
        print("[ERROR] No queries generated. Try relaxing constraints.")
        sys.exit(1)

    blip_caps = build_blip_captions(small_ds, device)
    print("Sample BLIP caption:", blip_caps[0])

    # -------------------------
    # Baseline-1: SigLIP Fusion
    # -------------------------
    metrics_fusion = eval_baseline_image_text_fusion(
        queries=queries,
        model=model,
        tokenizer=tokenizer,
        index=index,
        device=device,
        img_embeds_np=img_feats,
        alpha=FUSION_ALPHA,
        num_queries=NUM_Q_BASELINE,
        ks=KS,
        max_k=MAX_K,
    )
    print_metrics(f"Baseline-1: SigLIP Fusion (alpha={FUSION_ALPHA})", metrics_fusion, ks=KS, map_k=MAX_K)

    # -------------------------
    # Baseline-2: BLIP caption + concat_raw (no LLM)
    # -------------------------
    metrics_blip_concat = eval_baseline_blip_concat_raw(
        blip_caps=blip_caps,
        queries=queries,
        model=model,
        tokenizer=tokenizer,
        index=index,
        device=device,
        num_queries=NUM_Q_BASELINE,
        ks=KS,
        max_k=MAX_K,
    )
    print_metrics("Baseline-2: BLIP caption + concat_raw (no LLM)", metrics_blip_concat, ks=KS, map_k=MAX_K)

    # -------------------------
    # BLIP + LLM (multi-strategy)
    # -------------------------
    llm_modes = ["detail", "keywords", "conversational"]
    length_modes = [None, "short", "medium", "long"]

    for mode in llm_modes:
        for lm in length_modes:
            title = f"Ours: BLIP + LLM (mode={mode}, length={lm or 'default'})"
            metrics = eval_blip_llm(
                blip_caps=blip_caps,
                queries=queries,
                model=model,
                tokenizer=tokenizer,
                index=index,
                device=device,
                llm_mode=mode,
                length_mode=lm,
                num_queries=NUM_Q_OURS,
                ks=KS,
                max_k=MAX_K,
            )
            print_metrics(title, metrics, ks=KS, map_k=MAX_K)


if __name__ == "__main__":
    main()
