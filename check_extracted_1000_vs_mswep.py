"""
检查 extracted_basin_ids_1000 和 MSWEP 文件的流域是否匹配
"""

import pandas as pd

# 1. 读取候选列表前1000个流域（从extracted_basin_ids_1000.txt）
print("="*80)
print("检查 extracted_basin_ids_1000 vs MSWEP")
print("="*80)

print("\n[1] 读取 extracted_basin_ids_1000.txt...")
with open("extracted_basin_ids_1000.txt", 'r', encoding='utf-8') as f:
    content = f.read()

# 解析出流域列表
import ast
start = content.find("BASIN_IDS = [")
if start != -1:
    list_start = content.find('[', start)
    bracket_count = 0
    list_end = list_start
    for i in range(list_start, len(content)):
        if content[i] == '[':
            bracket_count += 1
        elif content[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                list_end = i + 1
                break
    
    list_str = content[list_start:list_end]
    extracted_1000 = ast.literal_eval(list_str)
    print(f"  提取的流域数量: {len(extracted_1000)}")
    print(f"  前10个: {extracted_1000[:10]}")
    print(f"  后10个: {extracted_1000[-10:]}")
else:
    print("  错误：无法找到BASIN_IDS")
    exit(1)

# 2. 读取MSWEP文件的流域列表
print("\n[2] 读取 MSWEP 文件...")
mswep_file = "MSWEP/mswep_1000basins_mean_3hourly_1980_2024.csv"
df_header = pd.read_csv(mswep_file, nrows=0)
mswep_basins = df_header.columns.tolist()[1:]  # 去掉第一列（时间）

print(f"  MSWEP包含的流域数量: {len(mswep_basins)}")
print(f"  前10个: {mswep_basins[:10]}")
print(f"  后10个: {mswep_basins[-10:]}")

# 3. 对比
print("\n[3] 对比分析...")

# 转换为字符串
extracted_1000_str = [str(b) for b in extracted_1000]
mswep_basins_str = [str(b) for b in mswep_basins]

# 匹配
matched = [b for b in extracted_1000_str if b in mswep_basins_str]
missing = [b for b in extracted_1000_str if b not in mswep_basins_str]

print(f"\n对比结果:")
print(f"  extracted_1000 数量: {len(extracted_1000)}")
print(f"  MSWEP 数量: {len(mswep_basins)}")
print(f"  匹配数量: {len(matched)} ({len(matched)/len(extracted_1000)*100:.1f}%)")
print(f"  缺失数量: {len(missing)} ({len(missing)/len(extracted_1000)*100:.1f}%)")

if len(matched) == len(extracted_1000):
    print("\n[OK] 完美！extracted_1000 中的所有流域都在MSWEP中")
else:
    print(f"\n[FAIL] 有 {len(missing)} 个流域不在MSWEP中")

# 4. 反向检查：MSWEP中有哪些不在extracted_1000中
mswep_not_in_extracted = [b for b in mswep_basins_str if b not in extracted_1000_str]
print(f"\n反向检查:")
print(f"  MSWEP中有但extracted_1000中没有的: {len(mswep_not_in_extracted)} 个")

# 5. 详细对比
print("\n" + "="*80)
print("详细分析")
print("="*80)

if len(extracted_1000) == len(mswep_basins):
    print("\n数量相同！都是1000个流域")
    
    # 检查是否完全一致（包括顺序）
    if extracted_1000_str == mswep_basins_str:
        print("[OK] 完全一致！顺序也相同")
    else:
        print("[WARN] 数量相同但顺序或内容不同")
        
        # 检查内容是否相同（不考虑顺序）
        if set(extracted_1000_str) == set(mswep_basins_str):
            print("[OK] 内容相同，只是顺序不同")
            
            # 找出顺序差异
            different_order = []
            for i in range(len(extracted_1000)):
                if extracted_1000_str[i] != mswep_basins_str[i]:
                    different_order.append({
                        'index': i,
                        'extracted': extracted_1000_str[i],
                        'mswep': mswep_basins_str[i]
                    })
            
            print(f"\n顺序不同的位置数量: {len(different_order)}")
            if len(different_order) <= 20:
                print("\n顺序差异详情:")
                for item in different_order[:20]:
                    print(f"  位置{item['index']}: extracted={item['extracted']}, mswep={item['mswep']}")
        else:
            print("[FAIL] 内容也不同")
            
            if missing:
                print(f"\nextracted_1000中有但MSWEP中没有的 ({len(missing)}个):")
                for b in missing[:20]:
                    print(f"  {b}")
                if len(missing) > 20:
                    print(f"  ... 还有 {len(missing)-20} 个")
            
            if mswep_not_in_extracted:
                print(f"\nMSWEP中有但extracted_1000中没有的 ({len(mswep_not_in_extracted)}个):")
                for b in mswep_not_in_extracted[:20]:
                    print(f"  {b}")
                if len(mswep_not_in_extracted) > 20:
                    print(f"  ... 还有 {len(mswep_not_in_extracted)-20} 个")
else:
    print(f"\n数量不同:")
    print(f"  extracted_1000: {len(extracted_1000)} 个")
    print(f"  MSWEP: {len(mswep_basins)} 个")
    print(f"  差值: {len(mswep_basins) - len(extracted_1000)} 个")
    
    if missing:
        print(f"\nextracted_1000中缺失的流域 (前20个):")
        for b in missing[:20]:
            print(f"  {b}")
        if len(missing) > 20:
            print(f"  ... 还有 {len(missing)-20} 个")
    
    if mswep_not_in_extracted:
        print(f"\nMSWEP中额外的流域 (前20个):")
        for b in mswep_not_in_extracted[:20]:
            print(f"  {b}")
        if len(mswep_not_in_extracted) > 20:
            print(f"  ... 还有 {len(mswep_not_in_extracted)-20} 个")

# 6. 结论
print("\n" + "="*80)
print("结论")
print("="*80)

if set(extracted_1000_str) == set(mswep_basins_str):
    print("\n[OK][OK][OK] 你是对的！")
    print("extracted_basin_ids_1000.txt 中的1000个流域")
    print("和 MSWEP 文件中的1000个流域")
    print("完全一致（内容相同）")
    
    if extracted_1000_str == mswep_basins_str:
        print("\n而且顺序也完全相同！")
    else:
        print("\n只是顺序可能不同")
else:
    print("\n[FAIL] 不完全一致")
    print(f"匹配: {len(matched)}/{len(extracted_1000)}")
    print(f"缺失: {len(missing)}/{len(extracted_1000)}")
