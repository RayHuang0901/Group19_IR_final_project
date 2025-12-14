import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from deepfashion_dataset import DeepFashionDataset  # 確保這支程式在同一目錄下
import csv
import os
from tqdm import tqdm

# --- 設定區 (Configuration) ---
DATA_PATH = "data/DeepFashion/img_highres/"
OUTPUT_FILE = "deepfashion_captions_optimized.csv"  # 檔名改一下，區分新舊結果
BATCH_SIZE = 32      # 如果顯卡記憶體不足 (OOM)，請改小 (例如 16 或 8)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型選擇：建議先用 base，如果顯卡夠強 (VRAM > 12G) 可以改用 "Salesforce/blip-image-captioning-large"
MODEL_NAME = "Salesforce/blip-image-captioning-base"

def extract_id_from_path(path):
    """
    從路徑解析 Item ID (用於後續計算 Recall)
    假設路徑結構: .../MEN/Denim/id_00000080/01_1_front.jpg
    目標提取: id_00000080
    """
    parts = path.split(os.sep)
    for part in parts:
        if part.startswith("id_"):
            return part
    return "unknown"

def main():
    print(f"=== DeepFashion Image Captioning Pipeline ===")
    print(f"使用裝置: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"輸出檔案: {OUTPUT_FILE}")

    # 1. 載入模型與處理器
    print(f"正在載入模型: {MODEL_NAME} ...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval() # 設定為評估模式

    # 2. 準備 Dataset
    # 注意：這裡不做 transform，因為 processor 會處理
    dataset = DeepFashionDataset(root_dir=DATA_PATH)
    
    # 定義 collate_fn 來處理 Prompt Engineering
    def collate_fn(batch):
        images = [item[0] for item in batch]
        paths = [item[1] for item in batch]
        
        # --- 關鍵優化 1: Prompt Engineering ---
        # 引導模型專注於細節 (材質、領口、風格)
        # 舊的 Prompt: "a photo of fashion clothing, "
        prompt_text = "a detailed photography of a fashion garment, identifying the fabric, texture, neck style and color: "
        
        text_prompts = [prompt_text] * len(images)
        
        # 轉成 Tensor
        inputs = processor(images=images, text=text_prompts, return_tensors="pt", padding=True)
        return inputs, paths

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 3. 開始推論並存檔
    print("開始生成圖片描述 (Inference)...")
    
    # 寫入 CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "item_id", "caption"]) # Header

        for inputs, paths in tqdm(dataloader):
            # 搬移數據到 GPU
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # --- 關鍵優化 2: 生成參數調整 (Inference-time Optimization) ---
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    # 長度控制
                    max_new_tokens=70,       # 允許生成較長的描述
                    min_length=25,           # 強制至少寫 25 個 token (逼它擠出細節)
                    
                    # 搜尋策略
                    num_beams=5,             # Beam Search: 搜尋前 5 種可能性，選最好的
                    
                    # 懲罰機制 (解決重複率 70% 的關鍵)
                    repetition_penalty=1.5,  # 嚴厲懲罰重複字詞 (設為 1.0 代表不懲罰)
                    
                    # 其他參數
                    length_penalty=1.0,      # 長度懲罰 (大於 1 鼓勵長句，小於 1 鼓勵短句)
                )

            # 解碼
            captions = processor.batch_decode(out, skip_special_tokens=True)

            # 寫入檔案
            for path, caption in zip(paths, captions):
                # 為了美觀，去除 Prompt 的部分文字 (選擇性，看你想不想留)
                # caption = caption.replace("a detailed photography of a fashion garment, identifying the fabric, texture, neck style and color: ", "")
                
                clean_path = path.replace(DATA_PATH, "")
                item_id = extract_id_from_path(path)
                
                writer.writerow([clean_path, item_id, caption.strip()])

    print(f"完成！結果已儲存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
