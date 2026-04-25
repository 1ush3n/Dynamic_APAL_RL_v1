import pandas as pd
import numpy as np
import argparse
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="生成基于原工序拓扑结构的随机扰动数据集 (支持拓扑剪枝防死锁)")
    parser.add_argument('--template', type=str, default='data/290.csv', help="原数据集骨架模板路径")
    parser.add_argument('--output_dir', type=str, default='data/random_datasets', help="输出目录")
    parser.add_argument('--num_samples', type=int, default=1, help="批量生成的样本数量")
    parser.add_argument('--drop_rate', type=float, default=0.15, help="删减真实工序的概率 (0.0~0.9)")
    parser.add_argument('--time_var', type=float, default=0.2, help="工时高斯波动的标准差系数 (如 0.2 代表上下浮动约 20%)")
    parser.add_argument('--seed', type=int, default=None, help="随机种子")
    return parser.parse_args()

def get_active_ancestors(node, drop_set, pred_map, memo, visited):
    """
    递归寻找依赖：如果直接前驱被删了，就越过它去找前驱的前驱。
    利用 memoization 避免重复计算，利用 visited 避免意外成环报错。
    """
    if node in visited:
        return set()
    visited.add(node)
    
    if node in memo:
        return memo[node]
        
    if node not in pred_map or not pred_map[node]:
        return {node} if node not in drop_set else set()
        
    active_preds = set()
    for p in pred_map[node]:
        if p in drop_set:
            active_preds.update(get_active_ancestors(p, drop_set, pred_map, memo, visited.copy()))
        else:
            active_preds.add(p)
            
    memo[node] = active_preds
    return active_preds

def generate_random_dataset(template_path, output_path, drop_rate, time_var):
    # 1. 读取模板 (统一作为字符串处理以防丢零)
    df = pd.read_csv(template_path, dtype=str)
    
    # 将需要的数值列转换回来
    df['类型'] = df['类型'].astype(int)
    df['加工时间/h'] = df['加工时间/h'].astype(float)
    
    # 2. 解析所有任务的依赖图
    pred_map = {}
    for idx, row in df.iterrows():
        node_id = str(row['AO号']).strip()
        preds_str = str(row.get('紧前工序AO号', ''))
        
        preds_list = []
        if pd.notna(preds_str) and preds_str.lower() not in ['nan', 'none', '', '0']:
            # 处理可能是带引号的多前驱，pandas 读 csv 会自动脱括号，只需要 split
            for p in re.split(r'[,，]', preds_str):
                p = p.strip()
                if p: preds_list.append(p)
                
        pred_map[node_id] = preds_list
        
    # 3. 决定抽杀哪些节点 (仅限 Type 2 真实工序)
    type2_indices = df[df['类型'] == 2].index
    num_to_drop = int(len(type2_indices) * drop_rate)
    
    drop_indices = np.random.choice(type2_indices, num_to_drop, replace=False)
    drop_set = set(df.loc[drop_indices, 'AO号'].str.strip())
    
    # 4. 依赖传递 (Bypassing)
    memo = {}
    for idx, row in df.iterrows():
        node_id = str(row['AO号']).strip()
        
        # 如果这个节点本身被删了，不需要为它更新前驱
        if node_id in drop_set:
            continue
            
        new_preds = set()
        for p in pred_map[node_id]:
            if p in drop_set:
                new_preds.update(get_active_ancestors(p, drop_set, pred_map, memo, set()))
            else:
                new_preds.add(p)
                
        # 更新回 DataFrame (用逗号拼接)
        if new_preds:
            df.at[idx, '紧前工序AO号'] = ','.join(sorted(list(new_preds)))
        else:
            df.at[idx, '紧前工序AO号'] = ''
            
    # 5. 正式剔除被删节点
    df = df.drop(drop_indices)
    
    # 6. 工时随机扰动 (仅对剩下的 Type 2)
    # New Duration = max(0.1, N(mu, (mu * time_var)^2))
    surviving_type2_mask = df['类型'] == 2
    mu = df.loc[surviving_type2_mask, '加工时间/h'].values
    sigma = mu * time_var
    
    new_durations = np.random.normal(loc=mu, scale=sigma)
    new_durations = np.maximum(0.1, new_durations) # 截断防负和防零 (0留给虚拟节点)
    
    # 保留一位或两位小数，显得更真实
    df.loc[surviving_type2_mask, '加工时间/h'] = np.round(new_durations, 2)
    
    # 7. 重排序号
    df = df.reset_index(drop=True)
    df['序号'] = df.index + 1
    
    # 确保保存时原本空的地方还是空的，不出现 nan 字符串
    df = df.fillna('')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return len(drop_set)

def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        
    if not os.path.exists(args.template):
        print(f"Error: 模板文件 {args.template} 不存在！请在项目根目录下运行。")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.template))[0]
    
    print(f"开始基于 {args.template} 生成随机数据集...")
    print(f"   - 预期生成数量: {args.num_samples}")
    print(f"   - 工序删减比例: {args.drop_rate * 100:.1f}%")
    print(f"   - 工时波动系数: {args.time_var * 100:.1f}%")
    print("-" * 50)
    
    for i in range(1, args.num_samples + 1):
        out_name = f"{base_name}_dr{args.drop_rate}_var{args.time_var}_s{i}.csv"
        out_path = os.path.join(args.output_dir, out_name)
        
        dropped_cnt = generate_random_dataset(args.template, out_path, args.drop_rate, args.time_var)
        print(f"[{i}/{args.num_samples}] 已生成: {out_name} (删减了 {dropped_cnt} 道工序)")
        
    print("-" * 50)
    print(f"全部完成！文件已存入 {args.output_dir}")

if __name__ == "__main__":
    main()
