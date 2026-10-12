# 新增困难流域：本地实验与属性审计

## 1. 实验对象、协议与可读取证据

### Takeaway
本次只读已有结果，不训练、不修改数据、模型或实验协议。99 流域由原 86 与 13 个真正新增站组成；14 候选中的 01100561 已在原 86 内。candidate_fixed 留出全部 14 候选，physical 回归拟合数为 85。[实验配置](D:/code/water_level_model_MSWEP/results/nonrating14_full_matrix/experiment.json)；[fixed 运行记录](D:/code/water_level_model_MSWEP/results/nonrating14_full_matrix/candidate_fixed_dual_head_ms1/run.json)

### Cited Findings
- 【协议事实】fixed 留出训练与验证 Q，保留 H；严格 random v2 的 protocol 为 train_valid_q_hidden_v2。random 必须逐运行读取 run.json 的 held_out_basins 后筛选 metrics.csv，不能将缺 mask_seed 的 metrics_all.csv 按 model_seed 直接聚合。[fixed 记录](D:/code/water_level_model_MSWEP/results/nonrating14_full_matrix/candidate_fixed_dual_head_ms1/run.json)；[random v2 配置](D:/code/water_level_model_MSWEP/results/nonrating99_random_physical_v2/experiment.json)
- 【代码事实】physical 使用 exp(intercept+slope×ln(area×p_mean)) 分别预测 Q 均值、标准差；observed 使用留出流域真实训练 Q 均值与标准差，因此是诊断条件，不能称严格 zeroQ。[normalization.py](D:/code/water_level_model_MSWEP/src/pipeline/normalization.py:112)；[ExternalPrepared](D:/code/water_level_model_MSWEP/src/evaluation/train_nonrating14_external.py:89)
- 【标签边界】“无当前公开 STGQ 曲线”仅是候选筛选标签，不能推定直接测量 Q、index-velocity 或确认非 rating。[候选清单](D:/code/water_level_model_MSWEP/data/coverage/nonrating_candidate_mswep_manifest.md)

### Inferences
- 本次所有机制结论均是本地数值支持的机制解释，不是已证实因果；外部站点地理核实由其他研究者负责。

### Gaps
- 当前执行环境没有 F: 盘，F:/data/CAMELSH/Hourly2/Hourly2/<ID>_hourly.nc 不可读。优先使用项目缓存；不为此重建完整数据管线。

### 已完成的审计口径与文件
- 【审计事实】本次读取 138 个运行：fixed 12 个、complete 18 个、严格 random v2 108 个。两种尺度的 random 各 54 个运行，30/50/70% × 两架构 × 三模型种子 × 三 mask_seed。逐 run.json 核验后，physical/observed 对应运行的真实留出 ID 完全相同；每站并非在全部三种 mask 中被留出。[运行审计清单](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_run_manifest.csv>)
- 【审计事实】保留 model_seed=1/2/3 与 mask_seed=42/123/456 的独立身份，只汇总真实留出 flow；complete 作为另列诊断。单任务/双头按站号、情景、尺度、模型种子、mask_seed 配对，582 对的测试 n 完全一致。[配对差值](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_paired_delta.csv>)
- 派生表保留每行源 metrics.csv 与 run.json 的绝对路径、真实留出标记和回归系数。三种子标准差采用样本标准差 ddof=1；未计算置信区间。mask 不同意味着训练流域组成不同，3/6/9 个运行分别代表 1/2/3 个留出 mask × 三模型种子，不能当作独立站点样本数。[逐运行审计](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_run_audit.csv>)；[分组汇总](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)

## 2. 核心站点表现：极负 NSE 是否主要来自均值与尺度偏差

### Takeaway
三种子均支持站点间失败形式不同：06843500 的 physical 预测量级崩溃且动态相关很弱；07230500 保留中等相关但幅度过大；02322800 与 06890900 的动态相关不足；05422600 对尺度方案的响应不同，不能归入统一的 physical 尺度错误解释。observed 是利用 Q 统计的诊断条件，其改善也不能证明单一因果。[三种子汇总](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)

### Cited Findings
- 以下均为 candidate_fixed、flow、三模型种子。每单元先逐种子计算，再取算术平均；NSE 的 ± 是三种子样本标准差。r 是预测 Q 与实测 Q 的 Pearson 相关，α=预测标准差/实测标准差，β=预测均值/实测均值；这两个比值是**测试预测表现分量，不是 physical 归一化回归的误差**。[逐种子原值与源路径](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_run_audit.csv>)

#### 六个重点站：fixed 三种子完整分量

|站号|尺度|架构|Q NSE 均值 ± SD|r 均值|α 均值|β 均值|
|---|---|---|---:|---:|---:|---:|
|06843500|physical|单任务|-4859.862 ± 3143.403|0.080|25.489|89.625|
|06843500|physical|双头|-8853.164 ± 9887.848|0.092|44.937|109.095|
|06843500|observed|单任务|-96.634 ± 83.697|0.095|2.754|11.732|
|06843500|observed|双头|-0.190 ± 0.114|0.143|0.502|1.327|
|07230500|physical|单任务|-218.887 ± 300.493|0.420|6.861|32.343|
|07230500|physical|双头|-17.177 ± 9.869|0.530|3.883|9.052|
|07230500|observed|单任务|-100.677 ± 66.111|0.264|6.557|23.215|
|07230500|observed|双头|-4.284 ± 1.319|0.567|2.615|3.047|
|07154500|physical|单任务|-572.807 ± 652.979|0.043|9.431|144.029|
|07154500|physical|双头|-62.303 ± 32.391|0.202|5.993|86.205|
|07154500|observed|单任务|-57.945 ± 59.738|0.049|4.636|87.298|
|07154500|observed|双头|-1.053 ± 0.582|0.238|0.766|-14.066|
|02322800|physical|单任务|-213.231 ± 224.762|0.155|10.902|3.571|
|02322800|physical|双头|-11.765 ± 5.253|0.264|3.353|1.402|
|02322800|observed|单任务|-10.292 ± 12.674|0.270|2.100|1.442|
|02322800|observed|双头|-1.113 ± 0.563|0.281|1.234|0.844|
|05422600|physical|单任务|-8.263 ± 6.100|0.288|2.503|4.691|
|05422600|physical|双头|-0.564 ± 0.583|0.525|1.114|2.495|
|05422600|observed|单任务|-34.966 ± 29.216|0.299|5.350|5.421|
|05422600|observed|双头|-1.120 ± 0.314|0.537|1.685|1.487|
|06890900|physical|单任务|-43.907 ± 62.565|0.027|4.356|10.177|
|06890900|physical|双头|-2.332 ± 0.409|0.217|1.475|3.583|
|06890900|observed|单任务|-72.099 ± 26.751|0.019|5.487|15.865|
|06890900|observed|双头|-0.761 ± 0.371|0.212|1.085|1.318|

表格全部来自 [local_candidate_summary.csv](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>) 的 candidate_fixed / flow 分组；各组 n_runs=3，种子 1/2/3。

#### NSE 分解：均值偏差能解释多少

用保存的指标直接验证恒等式：

`1−NSE = ((β−1)/CV_obs)² + (α−1)² + 2α(1−r)`，其中 `CV_obs=σ_obs/μ_obs`。

三项依次为均值偏差、幅度失配、去相关误差。表内占比是先逐种子计算再汇总的 `Σ该项/Σ(1−NSE)`，不是把平均 r/α/β 代入公式。全审计最大相对闭合误差约 6.56×10⁻⁶，误差来自保存指标的数值精度。去相关项带 α 权重，不能把其占比当作纯粹的因果贡献。[计算脚本](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/audit_local_results.py>)；[逐运行分解值](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_run_audit.csv>)

|站号|physical 双头测试 n|实测均值|实测标准差|CV|均值偏差占比|幅度失配占比|去相关占比|仅对齐测试均值后的诊断 NSE|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|06843500|5102|2.899830|4.355309|1.502|73.05%|26.05%|0.90%|-2384.987|
|07230500|9524|1.984519|6.924835|3.489|32.05%|47.95%|20.00%|-11.352|
|07154500|9456|0.131555|2.271541|17.267|39.39%|45.50%|15.11%|-37.367|
|02322800|9361|55.562199|19.372173|0.349|15.93%|46.12%|37.96%|-9.732|
|05422600|8980|0.939580|2.237875|2.382|28.47%|3.74%|67.79%|-0.119|
|06890900|9195|8.465381|25.073668|2.962|23.63%|7.21%|69.16%|-1.545|

均值/std 沿用 metrics 的 Q 数值单位，本次未重新核验 raw NC 单位元数据。n 是现成测试指标的有效预测/观测配对数，不能视为独立洪水事件数。仅对齐均值使用了测试真值，属于离线分解诊断，**不是可部署结果，也没有重新训练/修改预测**。[同一分组汇总源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)

#### 严格 random v2：真实留出后才汇总

每行 NSE 都只包含该站被留出的 mask。`n_runs/masks` 对每种尺度、每种架构相同；“—”表示该站在该比例下三个 mask 都没被留出，不能给它计算 zeroQ 分数。[真实留出清单](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_run_manifest.csv>)；[结果源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)

|站号|留出比例|n_runs/masks|physical 单任务 NSE|physical 双头 NSE|observed 单任务 NSE|observed 双头 NSE|
|---|---|---:|---:|---:|---:|---:|
|06843500|30%|6/2|-156.291|-1938.195|-0.377|-0.230|
|06843500|50%|3/1|-112709.197|-2390.064|-4.405|-0.160|
|06843500|70%|3/1|-769.170|-3794.422|-8.760|-0.160|
|07230500|30%|3/1|-0.490|-10.499|-0.260|-8.072|
|07230500|50%|6/2|-3.018|-5.600|-3.092|-4.371|
|07230500|70%|9/3|-4.811|-4.917|-10.633|-4.666|
|07154500|30%|0/0|—|—|—|—|
|07154500|50%|6/2|-95.175|-14.923|-1.728|-0.961|
|07154500|70%|6/2|-2.886|-11.740|-9.535|-0.643|
|02322800|30%|3/1|-15.244|-3.213|-0.642|-0.489|
|02322800|50%|6/2|-13.188|-5.317|-1.073|-0.734|
|02322800|70%|3/1|-46.337|-7.662|-12.548|-1.212|
|05422600|30%|3/1|-0.039|0.167|-0.671|-0.645|
|05422600|50%|6/2|-0.612|0.135|-2.037|-0.943|
|05422600|70%|6/2|-3.799|-0.139|-18.452|-1.271|
|06890900|30%|0/0|—|—|—|—|
|06890900|50%|6/2|-7.784|-0.818|-4.633|-1.048|
|06890900|70%|9/3|-7.259|-1.805|-6.905|-0.890|

严格 random 也重现 physical 的 06843500 量级崩溃，但双头并非所有站点/比例都胜过单任务：07230500 在 30%、50% 两种尺度均恶化；07154500 在 70% physical 下也恶化。比例之间的 mask、拟合尺度和供 Q 流域组成不同，不能把跨比例差异当作留出比例的单一因果效应。[同一结果源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)

#### 较好新增流域与辅助站对照

下表是 fixed 双头三种子的平均数；全部是真正新增站。较好对照说明“新增”“无当前公开 STGQ”均不足以决定差表现。[结果源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)

|站号|physical NSE ± SD|physical r/α/β|observed NSE ± SD|observed r/α/β|
|---|---:|---|---:|---|
|02407000|0.467 ± 0.026|0.740 / 0.937 / 1.171|0.447 ± 0.046|0.715 / 0.847 / 0.763|
|01189000|0.370 ± 0.048|0.641 / 0.471 / 0.829|0.414 ± 0.095|0.649 / 0.594 / 0.900|
|03198000|0.221 ± 0.370|0.766 / 1.266 / 1.186|0.280 ± 0.137|0.725 / 1.202 / 0.990|
|06893620|-0.305 ± 0.570|0.490 / 0.812 / 3.576|-0.477 ± 0.197|0.528 / 1.371 / 1.238|
|02365769|0.223 ± 0.019|0.526 / 0.644 / 1.340|-0.259 ± 0.168|0.525 / 1.244 / 1.175|

03198000 的 physical 结果种子离散较大，不宜包装为稳定成功。02365769 的 observed 反而变差，也反对“提供真实 Q 尺度总能解决问题”的统一解释。[同一源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)

#### 防止单个极端站扭曲总均值

13 个真正新增站、fixed 三种子先按站平均：physical 双头整体平均 NSE=-688.245、中位=-0.449，单任务平均=-458.000、中位=-8.263；observed 双头平均=-0.620、中位=-0.259，单任务平均=-32.023、中位=-13.073。physical 双头的总均值恶化由 06843500 极端值强烈驱动；不能据此声称全部新增站普遍恶化，也不能只报中位数隐去崩溃站。14 候选的口径另在 CSV 保留，避免混入重叠站 01100561。[分总体统计](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_population_summary.csv>)

### Inferences
- **06843500：支持量级崩溃与弱动态可迁移性并存。** physical 双头 β≈109、α≈45，73% 归一化误差是均值项，绝大部分其余误差是幅度项；observed 双头 NSE 回到 -0.19，但 r≈0.14 仍弱。physical 的去相关项占比小是因为巨大尺度误差压过其余项，绝不表示形状正确。仅对齐均值后的 NSE≈-2385，同样说明均值修正不够。physical fixed 双头种子 NSE 为 -3211.46/-20270.41/-3077.63，必须报告异常和不稳定性。[三种子/分解源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_run_audit.csv>)
- **07230500：支持部分动态捕捉但幅度、均值均失配。** 两尺度双头 r≈0.53–0.57，physical α≈3.88、β≈9.05，observed α≈2.61、β≈3.05；observed 下仍 NSE≈-4.28。不能称波动完全错误，也不能说有正确 Q 统计就能解决。random 30/50% 的双头恶化提醒存在情景依赖的负迁移/优化失配可能，因果未验证。[分组源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)
- **07154500：支持低均值、高 CV 使比值指标易变，同时幅度和动态失配。** 测试实测均值 0.1316、CV≈17.27，physical 双头均值项约39%、幅度项约45%，r≈0.20。observed β≈-14.07 表示预测均值为负，不能解读为已证明站点实测倒流；低均值使 β/PBIAS 极其敏感。高 CV 支持间歇/尖峰型流量分布的调查方向，但没有计算零流率，不能确认实际断流。[逐运行源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_run_audit.csv>)
- **02322800：支持幅度和动态错误为主。** physical 双头 r≈0.26、α≈3.35，均值项仅16%；observed 后 r≈0.28、β≈0.84，但 NSE 仍为负。其低 CV≈0.35 意味着同样相对均值误差在 NSE 分母较小的条件下影响更大，仍不能把失败统一归为均值尺度。需由外部站点笔记进一步支持可能的储蓄/地下水等机制，本地数值不能自行确认。[分解源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)
- **05422600：支持中等相关、偏差与动态误差共同作用，且 observed 不改善。** physical r≈0.53、α≈1.11、β≈2.50，observed r≈0.54、α≈1.68、β≈1.49，NSE 从 -0.56 降至 -1.12。均值更接近但幅度更偏离，不能用单一回归尺度问题解释。physical 三种子 NSE 为 0.091/-1.025/-0.759，不能据首个种子称稳定可用。[逐运行源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_run_audit.csv>)
- **06890900：支持动态迁移不足，尺度改善后仍弱。** physical/observed 双头 r 均约0.21；β 从3.58降到1.32、α从1.47降到1.08，observed NSE仍≈-0.76；observed误差约97%来自去相关项。这个站与07230500的中等相关站不同。[分解源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)
- 六站全标签 complete 双头也未形成普遍成功：physical/observed NSE 分别为 06843500 -46.461/-0.236、07230500 -0.692/-1.194、07154500 -0.324/0.036、02322800 -0.586/-0.154、05422600 -0.408/-0.717、06890900 -0.240/-0.292。支持“留出 Q 不是唯一问题”的机制解释；complete 的训练与尺度回归条件不同，不是单因素消融。[complete 分组源](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>)

### Gaps
- 未重跑训练、未生成模型预测时间序列。误差分解只证明数值结构，不证明洪峰时刻、局部事件机制或水文因果。observed 与 physical 是分别训练的模型，差异不能当作只替换反归一化尺度的受控实验。

## 3. 本地静态属性、Q 尺度与观测机制的证据和缺口

### Takeaway
已实读原 86 扩展属性 parquet（86×27）及诊断属性 CSV（86×19），六个重点新增站均不在其中，仅重叠站01100561有对应行。未找到99输入属性缓存，也未找到六站训练Q均值/std。因此能够确认测试预测量级问题，不能定量声称已算出这些站的 physical 回归尺度相对真实训练 Q 的误差。[86 属性缓存](D:/code/water_level_model_MSWEP/data/camelsh_exported/attributes_86_extended.parquet)；[诊断属性表](D:/code/water_level_model_MSWEP/data/coverage/basin_diagnostic_attributes.csv)；[原86训练统计](D:/code/water_level_model_MSWEP/data/splits/norm_stats.json)

### Cited Findings
- 【属性事实】原86扩展缓存含 area、p_mean、p_seasonality、frac_snow、aridity、slope_mean、frac_forest、土壤深度、湿地代理 swc_pc_syr、地质渗透性、土地覆被/地质 one-hot、高程、PET、土壤组/持水量、地下水位埋深、人口密度、urban_frac、dam_density、reservoir_storage 等共27输入列。六重点站以上属性全部缺少可读的实际输入行；不能给它们填气候干旱、土地覆被、水库或地下水具体值。[实际属性缓存](D:/code/water_level_model_MSWEP/data/camelsh_exported/attributes_86_extended.parquet)
- 【原86范围】area 10.3644–11983.3496；p_mean 2.629721–3.859869；p_seasonality -0.090349–0.237027；aridity 0.804130–1.224295；slope_mean 1.168305–26.959129；frac_forest 26.713707–97.975533；elev_mean 21.83671–799.59729；dam_density 0–23.149158；reservoir_storage 0–425.989685。除面积外本次沿用列原单位，未追读原始属性说明，不能擅自补单位或物理解读。以上是原86的范围，不是fixed回归85站范围。[同一属性缓存](D:/code/water_level_model_MSWEP/data/camelsh_exported/attributes_86_extended.parquet)
- 【本地站点元数据快照】以下面积来自已保存的 USGS 候选扫描表的 drain_sqkm，而非本次读取到的99模型输入属性；站名空值保持缺失。本次不做外部地理研究，不将缓存当作当前官网重核结果。[元数据快照](D:/code/water_level_model_MSWEP/results/summary/nonrating_basin_candidates_run1.csv)

|站号|本地缓存站名|drain_sqkm（km²）|相对原86面积范围|
|---|---|---:|---|
|06843500|Republican River at Cambridge, Nebr.|37413.1700|高于11983.3496上限，约3.12倍；元数据口径比较|
|07230500|缺失|1197.9380|范围内|
|07154500|缺失|3126.2510|范围内|
|02322800|SANTA FE RIVER NR HILDRETH FLA|3920.6280|范围内|
|05422600|Duck Creek at DC Golf Course at Davenport, IA|148.0923|范围内|
|06890900|DELAWARE R AT PERRY, KS|2923.2030|范围内|

表源：[候选元数据 CSV](D:/code/water_level_model_MSWEP/results/summary/nonrating_basin_candidates_run1.csv)；参照：[86属性 parquet](D:/code/water_level_model_MSWEP/data/camelsh_exported/attributes_86_extended.parquet)。06843500 的大面积支持检查空间尺度外推机制，但不证明输入area正确或尺度崩溃由面积单独造成。
- 【physical 回归事实】fixed dual_head seed1（其他fixed种子共享同一标签留出设定）记录均值斜率1.013958773、截距-5.053281442、训练拟合R²=0.986834151；std斜率0.932076004、截距-4.221068115、R²=0.953264553，n_fit=85。高拟合R²仅描述有Q的85站，不能保障新增站精度。[fixed run.json](D:/code/water_level_model_MSWEP/results/nonrating14_full_matrix/candidate_fixed_dual_head_ms1/run.json)
- 【缓存缺口核验】data/splits/norm_stats.json 的 flow 中六重点站均没有条目；zero_shot 两目录的 run.json 均未保存 stats/scaling_info。已有 zero_shot predictions parquet 只有 time/basin/q_observed/q_predicted，且属于另一实验条件，不拿它替代99训练统计或Q-H联合样本。[原86统计](D:/code/water_level_model_MSWEP/data/splits/norm_stats.json)；[旧外部推理记录](D:/code/water_level_model_MSWEP/results/zero_shot_nonrating14/run.json)；[旧推理缓存](D:/code/water_level_model_MSWEP/results/zero_shot_nonrating14/predictions_3hourly.parquet)

### Inferences
- 最强本地证据是测试预测的均值/幅度/动态误差分解以及observed诊断响应，外部站点事实应由其他研究者笔记补充后再逐站结合。不能据地区位置、候选标签或面积大小独自推导实际调控、地下水、湿地或率定方法。
- 原86支持域之外的属性机制目前只有06843500的**元数据面积**可做有限比较；其他新增站“面积范围内”不等于全部属性在训练支持域内。

### Gaps
- 六核心站及其他真正新增站的实际99输入属性缺失：p_mean、干旱/季节性、坡度/高程、覆被、湖泊/湿地、水库、人为调控、地下水/土壤等未完成站级量化。未计算相对fixed85多变量支持域的超范围程度。
- 缺少可读rawNC与99训练统计，**physical回归预测/真实训练Q均值和std的误差未能计算**；本文β/α绝不能冒充该误差。
- Q低流/零流比例、月份季节性、洪峰集中度、H-Q相关及滞回未验证。测试CV只是已有一二阶统计，不能代替这些机制检验。预测r也不等同H-Q相关。
- 本次只完成本地数值与缓存审计，没有新增外部站点事实、网页调查或原始论文归因；站点因果表述需与主协调者已取得的两份外部笔记结合，仍标为支持机制而非证实。

### 给报告写作者的核心结论
06843500应单列为严重量级崩溃且弱相关站；07230500应单列为有部分动态相关但幅度失配站；02322800/06890900重点是动态迁移不足；07154500需同时交代低均值高CV、比值敏感与负预测；05422600不能被“observed修好尺度”统一解释。较好02407000/01189000/03198000对照和random中的双头恶化例外都要保留。表述尺度原因时，准确说“支持尺度方案/量级失配机制”，不要说“已证实area×p_mean回归误差导致”。全部实验数值可从 [审计汇总](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_summary.csv>) 追溯到 [逐运行源路径](<D:/code/water_level_model_MSWEP/research_notes/新增困难流域 水文地理 成因调查/local_candidate_run_audit.csv>)。
