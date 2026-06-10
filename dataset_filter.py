import pandas as pd
import os
import math
from tqdm import tqdm


def preprocess_nyc_full():
    # ================= 1. 配置区域 (请根据您的 CSV 实际列名微调) =================
    input_path = 'dataset/NYC/NYC.csv'
    output_path = 'dataset/NYC/NYC_reprocessed.csv'

    # 核心列名 (根据 dataset.py 推断)
    col_user = 'user_id'
    col_poi = 'POI_id'
    col_tag = 'tag'

    # 时间列：脚本会尝试自动寻找 'timestamp' 或 'local_time'
    # 如果您的列名完全不同（比如叫 'Time'），请在这里修改
    preferred_time_col = 'timestamp'
    fallback_time_col = 'local_time'
    # =======================================================================

    # 检查文件
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到文件 {input_path}")
        return

    print(f"📂 正在读取文件: {input_path} ...")
    df = pd.read_csv(input_path)

    # --- 自动确定时间列 ---
    time_col = None
    if preferred_time_col in df.columns:
        time_col = preferred_time_col
    elif fallback_time_col in df.columns:
        time_col = fallback_time_col
    else:
        # 如果都找不到，尝试找任何包含 'time' 的列
        possible_cols = [c for c in df.columns if 'time' in c.lower()]
        if possible_cols:
            time_col = possible_cols[0]
            print(f"⚠️ 警告: 未找到标准时间列，尝试使用 '{time_col}' 作为时间依据。")
        else:
            print("❌ 错误: 无法确定时间列，无法进行排序。请修改脚本中的配置。")
            return

    print(f"🕒 使用时间列进行排序: {time_col}")

    # 确保时间列格式正确 (如果是字符串则转为datetime)
    if df[time_col].dtype == 'object':
        df[time_col] = pd.to_datetime(df[time_col])

    print("-" * 50)
    print(f"【原始数据】 行数: {len(df)}")
    print(f"   用户数: {df[col_user].nunique()}, POI数: {df[col_poi].nunique()}")
    print("-" * 50)

    # ================= 2. 过滤阶段 (Filter) =================

    # Step 1: 过滤掉历史总访问量 < 10 的 POI
    print("🧹 Step 1: 正在删除访问少于 10 次的 POI ...")
    poi_counts = df[col_poi].value_counts()
    valid_pois = poi_counts[poi_counts >= 10].index
    df = df[df[col_poi].isin(valid_pois)].copy()
    print(f"   -> 剩余行数: {len(df)}")

    # Step 2: 过滤掉历史总记录数 < 10 的 User
    # (注意：这是在 Step 1 删减后的基础上统计的，符合逻辑)
    print("🧹 Step 2: 正在删除记录少于 10 条的用户 ...")
    user_counts = df[col_user].value_counts()
    valid_users = user_counts[user_counts >= 10].index
    df = df[df[col_user].isin(valid_users)].copy()
    print(f"   -> 剩余行数: {len(df)}")

    # ================= 3. 重新划分 (Re-split 8:1:1) =================

    print("🔄 Step 3: 按时间顺序重新划分 Train/Val/Test (8:1:1) ...")

    # 核心操作：按 用户 和 时间 排序
    # 这是最关键的一步，保证了不会发生“未来数据泄露”
    df = df.sort_values(by=[col_user, time_col]).reset_index(drop=True)

    # 定义生成 Tag 的逻辑
    def assign_tags(group):
        n = len(group)
        # 计算分割点
        train_end = math.floor(n * 0.8)
        val_end = math.floor(n * 0.9)

        # 构造标签列表
        # list相加: ['train', 'train'] + ['val'] ...
        tags = ['train'] * train_end + \
               ['val'] * (val_end - train_end) + \
               ['test'] * (n - val_end)

        # 长度防抖（处理浮点精度导致的 +/- 1 误差）
        if len(tags) < n:
            tags += ['test'] * (n - len(tags))
        elif len(tags) > n:
            tags = tags[:n]

        return pd.Series(tags, index=group.index)

    # 应用划分逻辑 (使用 groupby + transform 保持 DataFrame 结构)
    tqdm.pandas(desc="正在生成标签")
    df[col_tag] = df.groupby(col_user)[col_user].transform(lambda x: assign_tags(x))

    # ================= 4. 保存结果 =================

    print("-" * 50)
    print("【最终结果统计】")
    print(f"   行数: {len(df)}")
    print(f"   用户数: {df[col_user].nunique()}, POI数: {df[col_poi].nunique()}")
    print("   数据集划分分布:")
    print(df[col_tag].value_counts())
    print("-" * 50)

    # 保存
    df.to_csv(output_path, index=False)
    print(f"✅ 成功! 文件已保存至: {output_path}")
    print(f"👉 提示: 请将 {output_path} 重命名为 NYC.csv 以替换旧数据。")


if __name__ == '__main__':
    preprocess_nyc_full()