# calculate_recall.py
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np

# --- 設定區 ---
CSV_FILE = "deepfashion_captions_with_id_version_01.csv"
IMG_ROOT = "data/DeepFashion/img_highres/"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 2000  # 測試用，跑正式數據時請設為 None

class CaptionDataset(Dataset):
    def __init__(self, csv_file, root_dir, sample_size=None):
        self.df = pd.read_csv(csv_file)
        # 確保路徑字串正確
        if 'image_path' not in self.df.columns:
            raise ValueError("CSV 檔案中找不到 'image_path' 欄位，請檢查上一生成的檔案。")

        if sample_size:
            self.df = self.df.head(sample_size)
            print(f"注意：目前僅使用前 {sample_size} 筆資料進行測試！")
        
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 處理路徑，確保沒有多餘的斜線
        rel_path = str(row['image_path']).lstrip('/')
        img_path = os.path.join(self.root_dir, rel_path)
        
        caption = str(row['caption'])
        item_id = str(row['item_id'])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # print(f"無法讀取圖片: {img_path}, Error: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0)) # 防呆黑圖

        return image, caption, item_id

def custom_collate_fn(batch):
    """
    這就是解決 TypeError 的關鍵：
    我們手動將 batch 裡的資料拆開，保持 image 為 list of PIL Images，
    而不是讓 PyTorch 嘗試把它變成 Tensor。
    """
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    item_ids = [item[2] for item in batch]
    return images, captions, item_ids

def compute_recall(features_a, features_b, labels_a, labels_b, k_list=[1, 5, 10]):
    # 正規化特徵向量 (很重要，否則 Cosine Similarity 會失準)
    features_a = features_a / features_a.norm(dim=-1, keepdim=True)
    features_b = features_b / features_b.norm(dim=-1, keepdim=True)

    # 計算相似度矩陣 [num_text, num_images]
    # 使用 GPU 計算會比較快，但如果記憶體不足會報錯
    # 這裡假設 sample size 不大，直接算
    sim_matrix = features_a @ features_b.T
    
    num_queries = len(labels_a)
    recalls = {k: 0 for k in k_list}
    
    print("正在計算 Recall 指標...")
    # 將 labels_b 轉為 numpy array 加速查找
    labels_b_arr = np.array(labels_b)
    
    for i in tqdm(range(num_queries)):
        scores = sim_matrix[i]
        
        # 找出前 K 個最高分的索引
        _, top_indices = scores.topk(max(k_list))
        top_indices = top_indices.cpu().numpy()
        
        # 取出對應的 ID
        retrieved_ids = labels_b_arr[top_indices]
        true_id = labels_a[i]
        
        for k in k_list:
            # 檢查正確答案是否在前 k 名中
            if true_id in retrieved_ids[:k]:
                recalls[k] += 1
                
    for k in k_list:
        recalls[k] = (recalls[k] / num_queries) * 100
        
    return recalls

def main():
    print(f"使用裝置: {DEVICE}")
    print("載入 CLIP (評估用模型)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = CaptionDataset(CSV_FILE, IMG_ROOT, SAMPLE_SIZE)
    
    # 這裡加入 collate_fn
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    all_text_embeds = []
    all_image_embeds = []
    all_ids = []
    
    print("開始提取特徵 (Feature Extraction)...")
    with torch.no_grad():
        for images, captions, item_ids in tqdm(dataloader):
            # 1. 處理文字
            text_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            text_features = model.get_text_features(**text_inputs)
            
            # 2. 處理圖片 (Processor 接受 List of PIL Images)
            img_inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
            image_features = model.get_image_features(**img_inputs)
            
            all_text_embeds.append(text_features.cpu())
            all_image_embeds.append(image_features.cpu())
            all_ids.extend(item_ids)
            
    # 合併 Tensor
    all_text_embeds = torch.cat(all_text_embeds).to(DEVICE)
    all_image_embeds = torch.cat(all_image_embeds).to(DEVICE)
    
    print(f"特徵提取完畢。矩陣大小: {all_text_embeds.shape}")
    
    # 計算 Recall
    recalls = compute_recall(all_text_embeds, all_image_embeds, all_ids, all_ids)
    
    print("\n========= 評估結果 (Image-to-Text Quality) =========")
    print(f"Total Samples Tested: {len(all_ids)}")
    print(f"Recall@1  : {recalls[1]:.2f}%")
    print(f"Recall@5  : {recalls[5]:.2f}%")
    print(f"Recall@10 : {recalls[10]:.2f}%")
    print("====================================================")

if __name__ == "__main__":
    main()
