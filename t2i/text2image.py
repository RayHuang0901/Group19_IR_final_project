import os
from typing import Optional, Dict, List, Tuple

from datasets import load_dataset
from openai import OpenAI
import torch
import open_clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import faiss
from collections import defaultdict
import random
import time


# ============================
# 全域 LLM client（避免每次重建）
# ============================
# 需先設好環境變數: OPENAI_API_KEY
LLM_CLIENT = OpenAI()
LLM_MODEL = "gpt-4o-mini"

# ============================
# LLM 相關工具函式
# ============================

def _length_instruction(length_mode: Optional[str]) -> str:
    """根據長度模式回傳一句放進 prompt 的說明文字。"""
    if length_mode is None:
        return ""
    if length_mode == "short":
        return "Limit the output to roughly 10-20 English words or fewer."
    if length_mode == "medium":
        return "Limit the output to roughly 40-50 English words."
    if length_mode == "long":
        return "You may use up to roughly 100 English words."
    # fallback
    return ""


def augment_caption(
    raw_desc: str,
    mode: str = "detail",
    length_mode: Optional[str] = None,
) -> str:
    """
    使用 LLM 對原始商品描述做 query rewriting。

    mode:
        - "detail"         : 詳細版自然語句描述（你原本的方法）
        - "keywords"       : 輸出逗號分隔的關鍵詞
        - "conversational" : 像使用者打在搜尋框的口語 query
    length_mode:
        - None / "short" / "medium" / "long"
    """
    length_instr = _length_instruction(length_mode)

    if mode == "detail":
        system_part = f"""
You are helping build a fashion image retrieval system.

Given the following raw product description, rewrite it into ONE natural English sentence
that only describes the visual appearance and style of the clothing item.

Rules:
- Focus on color, fit, cut, length, pattern, material type (e.g. denim, knit), and key design details (e.g. five-pocket, zip fly).
- Ignore washing instructions, size measurements, percentages, care labels, pricing, availability, and marketing slogans.
- Do NOT invent brands, people, or usage scenarios.
- The output should sound like a neutral product description, not an advertisement.
{length_instr}
""".strip()

    elif mode == "keywords":
        system_part = f"""
You are helping build a fashion image retrieval system.

Given the following raw product description, extract a comma-separated list of concise visual keywords.

Rules:
- Focus on color, fit, cut, length, pattern, material type (e.g. denim, knit), and key design details (e.g. five-pocket, zip fly).
- Ignore washing instructions, size measurements, percentages, care labels, pricing, availability, and marketing slogans.
- Do NOT invent brands, people, or usage scenarios.
- Return ONLY a comma-separated list of keywords, without any extra text.
{length_instr}
""".strip()

    elif mode == "conversational":
        system_part = f"""
You are helping build a fashion image search engine.

Given the following raw product description, rewrite it as a single conversational search query
that a user might type into a search bar to find this clothing item.

Rules:
- Focus on visual appearance and style (color, fit, cut, length, pattern, material, key design details).
- Do NOT invent brands, people, or usage scenarios that are not clearly implied.
- It should still be descriptive enough to match the product visually.
{length_instr}
""".strip()

    else:
        # 不認識的 mode 就退回 detail
        system_part = f"""
You are helping build a fashion image retrieval system.

Given the following raw product description, rewrite it into ONE natural English sentence
that only describes the visual appearance and style of the clothing item.
{length_instr}
""".strip()

    prompt = f"""
{system_part}

Raw description:
\"\"\"{raw_desc}\"\"\"

Output:
""".strip()


    

    for attempt in range(1,  6):
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
            else:
                raise RuntimeError("Empty LLM response")

        except Exception as e:
            last_err = e
            print(f"[LLM WARNING] attempt {attempt}/{5} failed: {e}")

            if attempt < 5:
                sleep_time = 2 ** (attempt - 1)
                print(f"[LLM] retrying after {sleep_time:.1f}s...")
                time.sleep(sleep_time)
    
    
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

# ============================
# Retrieval 主流程
# ============================

def run(model, tokenizer, preprocess, device, modes, length_modes):

    # 1. 載入資料 (subset)
    ds = load_dataset("Marqo/deepfashion-inshop", split="data")
    small_ds = build_small_ds_min15_per_triple(
        ds,
        total=5000,
        min_per_triple=15,
        min_colors_per_pair=2,
        seed=42,
    )
    print("Dataset size:", len(small_ds))
    print("Keys:", small_ds[0].keys())

    # 2. 計算 image embeddings
    print("Computing image embeddings...")
    img_feats = []

    for sample in tqdm(small_ds):
        img: Image.Image = sample["image"].convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        img_feats.append(feat.cpu().numpy())

    img_feats = np.vstack(img_feats).astype("float32")

    # 3. 建立 FAISS index
    print("Building FAISS index...")
    dim = img_feats.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(img_feats)

    # 4. 評測函式：Recall / Precision / MRR / mAP
    def compute_metrics(
        num_queries: int = 300,
        ks: Tuple[int, ...] = (1, 5, 10),
        llm_mode: Optional[str] = None,
        length_mode: Optional[str] = None,
        max_k: int = 50,
    ) -> Dict:
        num_queries = min(num_queries, len(small_ds))
        max_k = min(max_k, len(small_ds))

        # 累積用
        recall_sum = {k: 0.0 for k in ks}
        precision_sum = {k: 0.0 for k in ks}
        mrr_sum = 0.0
        map_sum = 0.0  # mean average precision

        for i in tqdm(range(num_queries)):
            raw_text = small_ds[i]["text"]

            # 1) 準備 query 文字
            if llm_mode is None:
                q = raw_text
            else:
                q = augment_caption(raw_text, mode=llm_mode, length_mode=length_mode)

            # 2) encode text
            with torch.no_grad():
                tokens = tokenizer([q]).to(device)
                feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy().astype("float32")

            # 3) 檢索 top max_k
            D, I = index.search(feat, max_k)
            retrieved_idx = I[0]  # 長度 max_k

            # 4) 定義 relevant：color + category1 + category2 都一樣
            gt_color = small_ds[i].get("color", None)
            gt_cat1 = small_ds[i].get("category1", None)
            gt_cat2 = small_ds[i].get("category2", None)

            relevance: List[bool] = []
            for idx in retrieved_idx:
                cand = small_ds[int(idx)]
                is_rel = (
                    cand.get("color", None) == gt_color
                    and cand.get("category1", None) == gt_cat1
                    and cand.get("category2", None) == gt_cat2
                )
                relevance.append(is_rel)

            num_rel = sum(relevance)

            # 5a) Recall@K & Precision@K
            for k in ks:
                topk_rel = relevance[:k]
                hit_count = sum(topk_rel)
                # Recall: 只要有任何一個 relevant 在前 k 就算 1
                recall_sum[k] += 1.0 if hit_count > 0 else 0.0
                # Precision: 前 k 裡 relevant 的比例
                precision_sum[k] += hit_count / float(k)

            # 5b) MRR
            if num_rel > 0:
                for rank, is_rel in enumerate(relevance):
                    if is_rel:
                        mrr_sum += 1.0 / float(rank + 1)
                        break
            # 否則對 MRR 貢獻 0

            # 5c) AP@m（m = max_k）
            if num_rel > 0:
                hit_cnt = 0
                ap = 0.0
                for rank, is_rel in enumerate(relevance):
                    if is_rel:
                        hit_cnt += 1
                        precision_at_rank = hit_cnt / float(rank + 1)
                        ap += precision_at_rank
                ap /= float(num_rel)
                map_sum += ap
            # 否則 AP 也是 0

        # 6) 平均成最終指標
        metrics = {
            "recall": {},
            "precision": {},
            "MRR": mrr_sum / num_queries,
            "mAP": map_sum / num_queries,
        }
        for k in ks:
            metrics["recall"][k] = recall_sum[k] / num_queries
            metrics["precision"][k] = precision_sum[k] / num_queries

        return metrics

    # 5. 實驗：不同 rewriting 策略 + 不同長度，多種指標

    for m in modes:
        for lm in length_modes:
            # baseline (m=None) 不需要掃不同長度，只跑一次
            if m is None and lm is not None:
                continue

            label_m = "baseline" if m is None else m
            label_l = "default" if lm is None else lm

            print(f"\n=== LLM mode: {label_m}, length: {label_l} ===")
            metrics = compute_metrics(
                num_queries=300,
                ks=(1, 5, 10),
                llm_mode=m,
                length_mode=lm,
                max_k=50,  # mAP@50
            )

            for k in (1, 5, 10):
                print(f"Recall@{k}:    {metrics['recall'][k]:.4f}")
                print(f"Precision@{k}: {metrics['precision'][k]:.4f}")
            print(f"MRR:            {metrics['MRR']:.4f}")
            print(f"mAP@50:         {metrics['mAP']:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 使用 Marqo 提供的 fashionSigLIP checkpoint
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:Marqo/marqo-fashionSigLIP"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")

    model.to(device)
    model.eval()

    modes = [None, "detail", "keywords", "conversational"]
    length_modes = [None, "short", "medium", "long"]
    run(model, tokenizer, preprocess, device, modes, length_modes)

    # 使用 Marqo 提供的 CLIP checkpoint
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model.to(device)
    model.eval()

    modes = [None]
    length_modes = [None]
    run(model, tokenizer, preprocess, device, modes, length_modes)


if __name__ == "__main__":
    main()
