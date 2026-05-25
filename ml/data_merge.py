"""
data_merge.py
=============
将 msd_subset.csv 与完整 musiXmatch BoW 歌词数据合并。

输入文件（放在 ../data/ 目录下）:
    msd_subset.csv          — MSD 10,000首歌的acoustic特征
    mxm_dataset_train.txt   — musiXmatch 训练集歌词 BoW
    mxm_dataset_test.txt    — musiXmatch 测试集歌词 BoW

输出文件:
    ../data/msd_mxm_merged.csv  — 合并后的完整数据

运行方式（在 ml/ 目录下）:
    python data_merge.py
"""

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, save_npz
import os
import re

# ─────────────────────────────────────────────
# 0. 路径配置
# ─────────────────────────────────────────────
DATA_DIR        = "../data"
MSD_PATH        = os.path.join(DATA_DIR, "msd_subset.csv")
MXM_TRAIN_PATH  = os.path.join(DATA_DIR, "mxm_dataset_train.txt")
MXM_TEST_PATH   = os.path.join(DATA_DIR, "mxm_dataset_test.txt")
OUTPUT_PATH     = os.path.join(DATA_DIR, "msd_mxm_merged.csv")
BOW_NPZ_PATH    = os.path.join(DATA_DIR, "mxm_bow_matrix.npz")  # 稀疏矩阵单独保存

# TF-IDF 保留的最大词数（从5000个词里选最有区分度的）
MAX_VOCAB = 1000


# ─────────────────────────────────────────────
# 1. 解析 musiXmatch .txt 文件
# ─────────────────────────────────────────────

def parse_mxm_file(filepath: str):
    """
    解析 mxm_dataset_train.txt 或 mxm_dataset_test.txt

    文件格式:
        # 注释行（以 # 开头）
        %word1,word2,...,word5000   ← 词表行（以 % 开头，只有一行）
        track_id,mxm_tid,is_test,word_idx:count,...  ← 数据行

    返回:
        vocab   : list of str，5000个词（按索引顺序）
        records : list of dict，每首歌的 {track_id, mxm_tid, is_test, word_counts}
                  word_counts 是 {词索引(1-based): 频次} 的字典
    """
    vocab = None
    records = []

    print(f"Parsing {os.path.basename(filepath)}...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 跳过注释
            if line.startswith("#"):
                continue

            # 词表行
            if line.startswith("%"):
                vocab = line[1:].split(",")
                print(f"  Vocabulary size: {len(vocab)}")
                continue

            # 数据行
            # 实际格式: track_id,mxm_tid,idx:count,idx:count,...
            # 注意: 没有 is_test 字段，train/test 由文件本身区分
            parts = line.split(",")
            if len(parts) < 2:
                continue

            track_id = parts[0]
            mxm_tid  = parts[1]

            # 解析稀疏词频: "idx:count" 格式，索引从1开始
            word_counts = {}
            for token in parts[2:]:
                token = token.strip()
                if ":" in token:
                    try:
                        idx, cnt = token.split(":")
                        word_counts[int(idx)] = int(cnt)
                    except ValueError:
                        continue

            records.append({
                "track_id":    track_id,
                "mxm_tid":     mxm_tid,
                "word_counts": word_counts,
            })

    print(f"  Songs parsed: {len(records)}")
    return vocab, records


# ─────────────────────────────────────────────
# 2. 合并 train + test，过滤出 MSD subset 的歌
# ─────────────────────────────────────────────

def build_lyrics_df(vocab, records, msd_track_ids: set, max_vocab=MAX_VOCAB):
    """
    把解析出的 records 转成 DataFrame，只保留 MSD subset 里的歌。

    同时做 TF-IDF 加权，选出最有区分度的 max_vocab 个词。

    返回:
        lyrics_df : DataFrame，index=track_id，columns=词，值=TF-IDF权重
        selected_words : list of str，选出的词
    """
    # 只保留 MSD subset 里有的歌
    filtered = [r for r in records if r["track_id"] in msd_track_ids]
    print(f"Songs matched with MSD subset: {len(filtered)} / {len(records)}")

    if len(filtered) == 0:
        raise ValueError("没有匹配到任何歌曲，请检查 track_id 格式是否一致")

    n_songs  = len(filtered)
    n_vocab  = len(vocab)
    track_ids = [r["track_id"] for r in filtered]

    # ── 2.1 构建原始词频矩阵（稀疏）────────────────
    print(f"Building word count matrix ({n_songs} x {n_vocab})...")
    count_matrix = lil_matrix((n_songs, n_vocab), dtype=np.float32)

    for i, record in enumerate(filtered):
        for word_idx, count in record["word_counts"].items():
            col = word_idx - 1  # 索引从1开始转为0开始
            if 0 <= col < n_vocab:
                count_matrix[i, col] = count

    count_matrix = count_matrix.tocsr()

    # ── 2.2 TF-IDF 加权 ──────────────────────────
    print("Computing TF-IDF...")
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_selection import SelectKBest, chi2

    tfidf_transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

    # 选出方差最大的 max_vocab 个词（用词频总和作为proxy）
    word_totals = np.asarray(count_matrix.sum(axis=0)).flatten()
    top_indices = np.argsort(word_totals)[::-1][:max_vocab]
    top_indices = np.sort(top_indices)  # 保持顺序

    selected_words = [vocab[i] for i in top_indices]
    tfidf_selected = tfidf_matrix[:, top_indices].toarray()

    print(f"Selected {len(selected_words)} words (from {n_vocab} total)")
    print(f"Top 20 words: {selected_words[:20]}")

    # ── 2.3 转成 DataFrame ────────────────────────
    col_names = [f"bow_{w}" for w in selected_words]
    lyrics_df = pd.DataFrame(
        tfidf_selected,
        index=track_ids,
        columns=col_names,
        dtype=np.float32,
    )
    lyrics_df.index.name = "track_id"

    return lyrics_df, selected_words


# ─────────────────────────────────────────────
# 3. 主流程
# ─────────────────────────────────────────────

def main():
    # ── Step 1: 读取 MSD subset ──────────────────
    print("=" * 50)
    print("Step 1: Loading MSD subset")
    msd_df = pd.read_csv(MSD_PATH)
    print(f"MSD subset shape: {msd_df.shape}")
    msd_track_ids = set(msd_df["track_id"].tolist())
    print(f"Unique track IDs: {len(msd_track_ids)}")

    # ── Step 2: 解析 musiXmatch 文件 ─────────────
    print("\n" + "=" * 50)
    print("Step 2: Parsing musiXmatch files")
    vocab_train, records_train = parse_mxm_file(MXM_TRAIN_PATH)
    vocab_test,  records_test  = parse_mxm_file(MXM_TEST_PATH)

    # 两个文件词表相同，合并records
    assert vocab_train == vocab_test, "Train/test vocab mismatch!"
    all_records = records_train + records_test
    print(f"Total musiXmatch songs: {len(all_records)}")

    # ── Step 3: 构建歌词特征矩阵 ─────────────────
    print("\n" + "=" * 50)
    print("Step 3: Building lyrics feature matrix")
    lyrics_df, selected_words = build_lyrics_df(
        vocab_train, all_records, msd_track_ids, max_vocab=MAX_VOCAB
    )

    # ── Step 4: 合并 MSD + 歌词 ──────────────────
    print("\n" + "=" * 50)
    print("Step 4: Merging MSD subset with lyrics")

    # left join：保留所有 MSD 歌曲，没有歌词的词列填 0
    merged_df = msd_df.merge(
        lyrics_df.reset_index(),
        on="track_id",
        how="left"
    )

    # 填充没有歌词的行为 0
    bow_cols = [f"bow_{w}" for w in selected_words]
    merged_df[bow_cols] = merged_df[bow_cols].fillna(0.0)

    # 添加 lyrics_available 标记
    has_lyrics = merged_df["track_id"].isin(
        set(lyrics_df.index)
    )
    merged_df["lyrics_available"] = has_lyrics.astype(int)

    print(f"Merged shape: {merged_df.shape}")
    print(f"Songs with lyrics: {merged_df['lyrics_available'].sum()} / {len(merged_df)}")
    print(f"BoW feature columns: {len(bow_cols)}")

    # ── Step 5: 保存 ──────────────────────────────
    print("\n" + "=" * 50)
    print("Step 5: Saving")
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")

    # 打印列名预览
    print(f"\n列结构预览:")
    print(f"  原始MSD列: {len(msd_df.columns)} 列")
    print(f"  新增BoW列: {len(bow_cols)} 列 (bow_<word>)")
    print(f"  总列数: {merged_df.shape[1]}")
    print(f"\n前5个BoW词: {selected_words[:5]}")
    print(f"后5个BoW词: {selected_words[-5:]}")

    # ── Step 6: 使用说明 ──────────────────────────
    print("\n" + "=" * 50)
    print("在 try2.py 中使用新数据:")
    print(f'  DATA_PATH = "../data/msd_mxm_merged.csv"')
    print(f"  歌词特征列名格式: bow_<word>，共 {len(bow_cols)} 列")
    print("""
  在 load_and_preprocess 里替换歌词部分:

    bow_cols = [c for c in df.columns if c.startswith('bow_')]
    lyric_matrix = df[bow_cols].values.astype(np.float32)
    # 已经是TF-IDF加权，不需要再做TfidfVectorizer
    # lyrics_available 列已存在，直接用
    has_lyrics_mask = df['lyrics_available'] == 1
    """)


if __name__ == "__main__":
    main()