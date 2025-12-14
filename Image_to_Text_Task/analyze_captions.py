# analyze_captions.py
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# 讀取你的 CSV
CSV_FILE = "deepfashion_captions_optimized.csv"
df = pd.read_csv(CSV_FILE)

# 1. 計算句子長度 (單字數)
df['word_count'] = df['caption'].apply(lambda x: len(str(x).split()))

# 繪製長度分佈圖
plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Caption Lengths (Word Count)')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.axvline(df['word_count'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {df["word_count"].mean():.2f}')
plt.legend()
plt.savefig('caption_length_distribution.png')
print(f"平均句子長度: {df['word_count'].mean():.2f} 個單字")

# 2. 詞彙分析 (看看模型都說了什麼)
all_text = " ".join(df['caption'].astype(str).tolist()).lower()
# 移除標點符號
all_text = re.sub(r'[^\w\s]', '', all_text)
words = all_text.split()

# 統計最常出現的前 20 個字
word_counts = Counter(words)
common_words = word_counts.most_common(20)

print("\n=== 最常出現的 20 個詞彙 ===")
for word, count in common_words:
    print(f"{word}: {count}")

# 3. 檢查「多樣性」(有多少 unique 的描述)
unique_captions = df['caption'].nunique()
total_captions = len(df)
print(f"\n=== 多樣性分析 ===")
print(f"總共有 {total_captions} 筆資料")
print(f"只有 {unique_captions} 種不重複的描述")
print(f"重複率: {(1 - unique_captions/total_captions)*100:.2f}%")
