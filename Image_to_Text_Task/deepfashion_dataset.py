# deepfashion_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset

class DeepFashionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 圖片的最上層目錄 (例如 data/DeepFashion/img_highres/)
            transform (callable, optional): PyTorch 的 transform 操作
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # 支援的圖片格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        print(f"正在掃描資料夾: {root_dir} ...")
        # os.walk 會遞迴進入 MEN, WOMEN 以及其子資料夾
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"掃描完成！共找到 {len(self.image_paths)} 張有效圖片。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # convert('RGB') 確保圖片都是 3 通道，避免灰階或 RGBA 報錯
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"無法讀取圖片 {img_path}: {e}")
            # 如果讀取失敗，回傳一個全黑的圖或者是 None (視後續處理而定)
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        
        # 回傳圖片本身以及它的路徑（方便你知道這張圖是誰）
        return image, img_path
