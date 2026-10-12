# 新增困难流域的水文地理与可能误差机制：02322800、05422600、06890900

## 问题一：这些站点是谁，本次实验比较能够说明什么？

### Takeaway

三站分别是 Florida 的 Santa Fe River near Hildreth、Iowa 的 Duck Creek at Davenport 和 Kansas 的 Delaware River at Perry，不能用同一种水文机制解释其差表现。02322800 的巨大改善首先支持检查均值与尺度；06890900 有直接的调度和回水证据；05422600 换用 observed 尺度后反而下降，需要保留不同解释。

### Cited Findings

调查日期：2026-10-10。仅检索外部公开资料并保存本笔记；未训练、未准备数据、未改变数据、模型或实验协议。以下标签必须保留：**站点事实**指明确对应站号的官方记录；**流域/区域事实**指较大范围或历史调查，不能冒充本站实测属性；**机制推断**只说明与现象相容；**任务给定/待核验**指协调者提供、由另一研究者核实的实验信息。

官方身份按站号查询，坐标为 USGS 标注的 WGS84 十进制度、展示到六位小数；面积保留官方平方英里，未从站名推断位置。

| 站号 | 官方站名 | 位置与水系 | WGS84 纬度、经度 | 官方排水面积 | 支持事实的官方来源 |
| --- | --- | --- | --- | --- | --- |
| 02322800 | SANTA FE RIVER NR HILDRETH FLA | Florida，Gilchrist County；Santa Fe，HUC 03110206 | 29.911621，-82.860403 | 1,374 mi² | [USGS 站点元数据](https://waterdata.usgs.gov/monitoring-location/USGS-02322800/) |
| 05422600 | Duck Creek at DC Golf Course at Davenport, IA | Iowa，Scott County；Copperas-Duck，HUC 07080101 | 41.546114，-90.524002 | 57.3 mi² | [USGS 站点元数据](https://waterdata.usgs.gov/monitoring-location/USGS-05422600/)；[站点说明与面积](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=05422600) |
| 06890900 | DELAWARE R AT PERRY, KS | Kansas，Jefferson County；Delaware，HUC 10270103；邻近 Kansas River 汇流处 | 39.075000，-95.403863 | 1,141 mi² | [USGS 站点元数据](https://waterdata.usgs.gov/monitoring-location/USGS-06890900/)；[水年说明](https://nwis.waterdata.usgs.gov/ks/nwis/wys_rpt/?site_no=06890900) |

**任务给定/待另一研究者核验，非本次独立计算的实验结果：**原 86 流域多在美国东北；14 个候选中 01100561 与原 86 重叠，真正新增 13 个，总计 99 个。fixed holdout 在训练和验证中屏蔽目标流域 Q 标签、保留 H；模型输入为 MSWEP 降雨、气温、太阳辐射与 static，没有历史 Q 和人类活动动态。physical 的 Q 均值/std 通过 area × p_mean 的回归预测，未使用目标流域真实 Q 尺度；observed 使用留出流域真实训练期 Q 均值/std，属于诊断条件，不是严格 zero-Q。以上来源为本轮任务说明，未冒充外部文献或已核验代码。

| 站号 | candidate_fixed 双头 physical NSE | 双头 observed NSE | Δ = observed − physical | 数值的直接含义，待结果核验 |
| --- | ---: | ---: | ---: | --- |
| 02322800 | -11.765 | -1.113 | +10.652 | 对尺度条件非常敏感，但 observed 仍低于 NSE 的观测均值参照 |
| 05422600 | -0.564 | -1.120 | -0.556 | 真实训练 Q 尺度没有改善该比较 |
| 06890900 | -2.332 | -0.761 | +1.571 | 尺度条件可能贡献部分误差，仍有剩余困难 |
| 06893620 | -0.306 | -0.477 | -0.171 | 辅助站；不能统一归因为 physical 尺度偏差 |
| 02365769 | 0.223 | -0.259 | -0.482 | 辅助站；physical 在该比较中更好 |

NSE 的观测均值参照及误差分解来自原始论文，不是“负 NSE 就必然没有学到波动”的规则。[Gupta et al., 2009，原始论文出版页](https://www.sciencedirect.com/science/article/pii/S0022169409004843)；[原论文全文](https://ag.purdue.edu/hydrology/wiki/environmental-informatics/modeling-resources/attachments/gupta-et-al-2009.pdf)

### Inferences

- **机制推断：**按任务给定的两列 NSE，02322800 应优先检查训练尺度预测偏差和测试期均值偏差；不能仅凭 -11.765 宣称波动形状完全错误。06890900 可同时存在尺度问题和受控河流动态问题。05422600 与两辅助站的下降则反驳“所有新站都只需修正尺度”的统一解释。推断依据为上表任务给定值及 [NSE 分解原始论文](https://www.sciencedirect.com/science/article/pii/S0022169409004843)。
- **比较边界：**physical→observed 不应自动视为给同一条预测曲线乘系数；未核实两条件是否分别训练、损失归一化是否变化，以及随机种子是否配对。因此改善不能直接量化为“纯反归一化误差被消除”。这是对任务信息的逻辑限制，不是代码审计结论。

### Gaps

- 本次依照“仅外部调查”的分工未读取实验指标文件、实际 static 向量或逐时 Q/H/预测；不是声称这些文件无法读取。缺少模型/遮蔽种子、聚合方式、测试有效 n、具体时段及两条件是否配对，实验表暂以任务给定值为准。
- 无本站实际均值/std、预测相关系数、峰值时序误差及 physical 尺度回归残差，尚不能判断哪个误差分量占比最大。未启动 pipeline 或全量数据准备。
- “无当前公开 STGQ 曲线”只能说明某次当前曲线检索的结果，不等于直接测量 Q，也不等于确认非 rating。本次未重新执行 STGQ 扫描。06890900 的特殊算法有独立站点记录，不能外推到其他候选；06893620 官方页面还保留了水位—流量曲线入口，见问题二。

## 问题二：三站有哪些直接水文证据，水位是否可能受下游控制？

### Takeaway

证据最强的是 06890900：USGS 明确记载完全受水库调节、流速指数法及回水造成负流速/负流量。02322800 的官方站页有潮汐/风影响提示，原始 USGS 研究另有 Santa Fe 下游受 Suwannee 洪水顶托的记录；05422600 已确认农业与城市混合流域背景，但没有找到其测站受 Mississippi 回水、湖库或潮汐控制的明确记录。

### Cited Findings

#### 02322800：Santa Fe River near Hildreth，Florida

- **站点事实，位置：**历史 USGS 记录明确把 02322800 定位在 US Highway 129 桥、Santa Fe 河口上游 2.4 mi。这里接近 Santa Fe 与 Suwannee 汇流处；不是凭站名猜测。[USGS Florida 水年 1990，杂项站测量表中的 02322800](https://pubs.usgs.gov/wdr/1990/fl-90-4/report.pdf)
- **站点事实，特殊影响提示：**官方站点 remarks 提示计算流量可能因潮汐影响、风或其他因素变化而不同于单次测量。这支持把下游边界影响列入候选机制，但页面没有量化潮汐幅度、频率或模型测试期影响。[USGS 02322800 站点说明](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=02322800)
- **站点事实，方法时间边界：**当前官方页面列出参数 72255，即用于流量计算的平均流速，显示数据开始于 2026 年 5 月。仅能确认当前页面存在这一变量，不能据此判定 2001–2024 或本实验实际训练期采用何种流量计算方法，更不能因 STGQ 缺失宣称历史 Q 是直接测量。[USGS 参数列表](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=02322800)
- **下游河段事实，回水与蓄存：**1983 年 USGS 原始研究记载：Santa Fe 下游可受 Suwannee 洪水顶托，洪水期下游峰值流量受到压低，部分洪水留存至 Suwannee 水位回落；下游洪泛平原许多位置低于海拔 20 ft。该记录针对下游河段，不是本站逐时回水标记。[Hunn & Slack, 1983，Water Resources of the Santa Fe River Basin，正文印刷页 50](https://pubs.usgs.gov/wri/1983/4075/report.pdf)
- **流域事实，岩溶与降雨：**同一原始研究摘要说明，Santa Fe 流域西部为岩溶平原，Floridan 石灰岩含水层的隔水条件弱；东部隔水条件较强。历史流域年均降雨约 54 in。这些支持地下水—河流交换与空间异质性，不能作为本站最新坡度、泉水贡献率或 CAMELSH 属性值。[Hunn & Slack, 1983，摘要](https://pubs.usgs.gov/wri/1983/4075/report.pdf)

#### 05422600：Duck Creek at DC Golf Course at Davenport，Iowa

- **站点事实，位置和仪器：**USGS 说明其位于 Kimberly Road 上游 500 ft、Pheasant Creek 汇入口下游 0.5 mi、Duck Creek 河口上游 4.45 mi，面积 57.3 mi²，使用水位记录仪。仪器类型不是流量生成算法的完整说明。[USGS 05422600 站点说明](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=05422600)
- **同一完整流域事实，土地利用和边界：**Iowa DNR 收录的 2011 管理计划说明，上游西半部偏农业、下游东半部偏城市，Duck Creek 直接汇入 Mississippi。完整流域约 64 mi²；表 3.1 中农业为 46%、两类住宅合计约 21%。这些是完整流域历史土地利用，不能直接填成面积 57.3 mi² 测站上游的属性。[Duck Creek Watershed Management Plan 2011，印刷页 6、9–11](https://www.iowadnr.gov/media/4192/download?inline=)
- **完整流域事实，地形和人工排水：**计划描述基岩露头、陡侧坡、狭窄谷底与黄土覆盖的冰碛平原；另记载河道渠化、暗管、雨水管网与人工衬砌通道。不能据此称“全流域平均坡度很陡”，也没有测站上游各类排水设施比例。[同一计划，印刷页 17–20](https://www.iowadnr.gov/media/4192/download?inline=)
- **县域气候事实：**计划将 Scott County 归为湿润大陆性气候，列出历史年降雨 33.42 in、年降雪 30.6 in。降雪深度不能直接加到降雨量当作水当量；该资料也不是本站逐月降雨正常值。[同一计划，印刷页 20](https://www.iowadnr.gov/media/4192/download?inline=)
- **站点事实，洪水响应：**USGS 的 2008 年中西部洪水原始报告在本站行列出 2008-06-13 峰值流量 7,570 ft³/s、面积 57.3 mi²。这确认本站发生过显著洪峰，但单次峰值本身不证明响应时间短于 3 h，也不证明峰值由回水造成。[USGS Professional Paper 1775，峰值表中 05422600](https://pubs.usgs.gov/pp/1775/pdf/pp1775.pdf)

#### 06890900：Delaware River at Perry，Kansas

- **站点事实，受控程度与算法：**USGS 2024 水年说明明确记载流量自 1969 年以来完全由 Perry Lake 调节；2003-05-05 后采用流速指数法计算流量，测站使用声学流速记录仪。[USGS 06890900 水年说明，GAGE/REMARKS](https://nwis.waterdata.usgs.gov/ks/nwis/wys_rpt/?site_no=06890900)
- **站点事实，下游回水：**同一说明明确其接近 Kansas River 汇流处，计算流量有时受回水影响；回水时连续测得的流速可为负，计算流量因而也可为负。负 Q 在这里有官方物理/计算依据，不能默认是坏值。[同一说明，REMARKS](https://nwis.waterdata.usgs.gov/ks/nwis/wys_rpt/?site_no=06890900)
- **站点事实，历史位置变化：**2003-05-05 前位置在现站上游 4.4 mi，历史名称为 below Perry Dam；早期流量由库水位和闸门出流结构计算。若实验时段跨越迁站，站号一致不保证测量条件一致。官方目前列总面积 1,141 mi²、probably noncontributing 面积 1,116.83 mi²；元数据另列 contributing area 24.17 mi²，不能把后者当成总面积。[同一说明，LOCATION/DRAINAGE AREA/GAGE](https://nwis.waterdata.usgs.gov/ks/nwis/wys_rpt/?site_no=06890900)；[当前元数据](https://waterdata.usgs.gov/monitoring-location/USGS-06890900/)
- **库区事实，调度行为：**USACE Perry Lake 项目介绍说明其用于防洪、蓄水及休闲，工作人员协调改变对 Delaware River 的放水。该资料支持人为动态确实存在，未提供本实验期间逐时调度序列。[USACE Perry Lake 项目介绍](https://www.nwk.usace.army.mil/Locations/District-Lakes/Perry-Lake/)
- **上游流域历史事实，土地利用：**USGS 1996 原始调查把 Delaware River Basin 约 85% 列为农业，其中约 40% 为作物，其他农业土地为牧场/草场。这不是当前本站 static 值，也不意味着本站出流与当场降雨同步。[USGS OFR 96-331，Land Use](https://pubs.usgs.gov/of/1996/0331/report.pdf)

#### 辅助站：仅保留足够可靠的信息，不扩展成完整归因

- **06893620，站点事实：**官方名 Rock Creek at Kentucky Road in Independence, MO；Jackson County，HUC 10300101；WGS84 39.111944、-94.472222，面积 9.5 mi²。当前站页曾提示 2024-10-29 向上游约 0.25 mi 迁至 06893619；如果涉及实验末期，需核查是否混用了两站记录，不能先判定项目已经混用。[USGS 06893620 元数据与站点通知](https://waterdata.usgs.gov/monitoring-location/USGS-06893620/)
- **06893620，分类边界：**官方 legacy 页面显示水位—流量曲线入口，并提示冰冻可影响水位和流量。这反映官方页面曾提供曲线线索，不等于本次已读到 STGQ 文件；也不能把候选标签解释为确认为非 rating。[USGS 06893620 站点说明](https://waterdata.usgs.gov/nwis/uv/?referred_module=sw&site_no=06893620)
- **02365769，站点事实：**官方名 BRUCE CREEK AT SH 81 NR REDBAY, FL；Walton County，WGS84 30.624636、-85.942436。2001 水年报告明确 HUC 03140203、面积 82.4 mi²、桥下游侧测站与水位记录仪，并说明曾修订水年 2000 日/月流量和极值。这是历史数据质量背景，不能据此认定本实验数据损坏或该站受潮汐。[USGS 当前元数据](https://waterdata.usgs.gov/monitoring-location/USGS-02365769/)；[USGS Florida 水年 2001，02365769](https://fl.water.usgs.gov/PDF_files/dataRptFL01_4.pdf)

### Inferences

- **02322800，支持机制而非已证实原因：**地下水/岩溶交换可能使地表 area × p_mean 难以表征真实 Q 的均值和方差；下游顶托可能抬高 H 而不同比例增加向下游 Q。因此，气象与 static 驱动的模型既可能出现尺度问题，也可能面对训练域之外的 H–Q 关系。证据为 [Santa Fe 原始水资源调查](https://pubs.usgs.gov/wri/1983/4075/report.pdf) 和 [本站 remarks](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=02322800)，尚无本次预测误差的事件对应证据。
- **05422600，支持机制而非已证实原因：**农业—城市混合和人工排水可能使局地暴雨、产汇流时间及土地利用变化更难由 3 h 流域平均降雨与固定属性表达。可能影响峰值时序/幅度，但不能先断言 MSWEP 低估了本站暴雨或缺少了具体属性。[Duck Creek 原始管理计划](https://www.iowadnr.gov/media/4192/download?inline=)
- **06890900，支持机制而非已证实原因：**库水位、闸门调度和 Kansas River 下游水位决定的动态没有包含在任务给定输入中。H 监督仍可含有有用信息，但它所反映的下游控制不必对应本地降雨产生的 Q；因此不能保证多任务共享表征必然改善 Q。这一假设有强站点依据，未完成本实验因果验证。[USGS 调度、算法和回水记录](https://nwis.waterdata.usgs.gov/ks/nwis/wys_rpt/?site_no=06890900)

### Gaps

- **02322800：**未核实训练/测试期流量算法、回水/潮汐发生频率和幅度、地下水贡献率、地下水补给区边界、土地覆被/湖泊/湿地比例、本站平均坡度、实际取水与调度历史。官方 remarks 的潮汐/风提示不能单独证明本站一直是典型潮汐河，也不能用 2026 平均流速变量反推历史算法。
- **05422600：**未找到明确对应 05422600 的 Mississippi 顶托、湖库控制或特殊流量估算法记录；距河口 4.45 mi 和汇入 Mississippi 都不足以证明本站 H 受其回水。未量化本站上游城市不透水率、湖泊/湿地、地下水/岩溶、取水、平均坡度和逐月降雨；不能用 Wisconsin、Alaska、Texas 或 Las Vegas 的同名 Duck Creek 研究代替本站资料。
- **06890900：**未核实本站平均坡度、降雨季节分配、当前土地利用、地下水/岩溶/湿地属性、实际取水量和训练/测试期调度过程。原始东北流域与受控 Kansas 河流的差异是分布差异线索，不是已完成的属性对比。
- **检索质量：**Iowa 管理计划及 Gupta 原论文可读取正文；若干 USGS 历史 PDF、水年动态页和 USACE 页的直接打开返回 Internal Error/403，本笔记相应事实依据官方页面的检索缓存正文/段落。未声称完整读通这些受限文件；历史事实均保留年份。对水年说明的关键算法与回水信息，检索返回的是同一官方页面的完整说明段落。PDF 图像核查亦有失败，因此不依赖未看清的地图或图形定量取值。
- **辅助站：**仅完成身份、面积及少量方法/迁站线索；气候、地形、水库、顶托、湿地和城市化比例仍待验证。02365769 带与不带 USGS- 前缀的网页缓存曾呈现不同数据可用性，不把“页面暂无图”解释成历史 Q/H 不存在。

## 问题三：如何区分均值/尺度误差与波动形状误差，给协调者什么结论？

### Takeaway

最可靠的归因层次是“本站有何官方过程证据、该过程能支持何种模型机制、还缺哪项诊断”，而不是把 NSE 单值当作机制证明。02322800 优先查尺度与均值，06890900 同时保留调度/回水和尺度问题，05422600 优先保留事件响应与输入表达等机制；均未达到已证实因果的程度。

### Cited Findings

**可直接复用的诊断分解：**对同一测试有效样本集合，定义 r 为预测与观测的 Pearson 相关系数，α = σ预测 / σ观测，b = (μ预测 − μ观测) / σ观测。使用相同样本和分母为 n 的均值/方差，且 σ观测 > 0、相关系数有定义时：

```text
NSE = 2rα − α² − b²
    = r² − (α − r)² − b²
```

这将相关性、条件偏差与均值偏差分开，不能仅从负 NSE 判定形状差。分解见原论文印刷页 81、式 (3)–(4)；这里用 b 表示按观测标准差归一化的均值偏差，避免与 KGE 常用均值比符号混淆。[Gupta et al., 2009 原论文全文](https://ag.purdue.edu/hydrology/wiki/environmental-informatics/modeling-resources/attachments/gupta-et-al-2009.pdf)

**解释用算例，非本站实测：**若 r = 1、α = 1、b = 3，即波动完全同步且标准差相同，只差 3 个观测标准差的均值，则 NSE = -8。非常负的 NSE 完全可能包含较好的波动形状；反过来 observed 仍为负也不能仅据此证明相关性低。这是上式的代数示例。[公式来源](https://www.sciencedirect.com/science/article/pii/S0022169409004843)

| 站点 | 已确认的最强事实 | 与任务给定数值相容的机制，均未证实因果 | 目前不能说的话 |
| --- | --- | --- | --- |
| 02322800 | 下游回水研究、本站潮汐/风提示、岩溶背景；[USGS 研究](https://pubs.usgs.gov/wri/1983/4075/report.pdf)、[本站说明](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=02322800) | +10.652 的改善提示尺度条件敏感；岩溶交换可能影响 Q 尺度；回水可能使 H 与向下游 Q 的关系变化 | “主要误差已证明全是尺度”“形状完全错”“历史一定采用流速法”“一定是潮汐流域” |
| 05422600 | 城乡混合和人工排水背景；[Iowa 原始计划](https://www.iowadnr.gov/media/4192/download?inline=) | observed 下降，与“修好尺度就解决”不符；局地事件响应、时序与输入表达值得检查 | “Mississippi 已证实顶托该站”“NSE 差必由城市化/降雨误差造成” |
| 06890900 | 水库完全调节、流速指数法、Kansas River 回水和负流速/负 Q；[USGS 水年说明](https://nwis.waterdata.usgs.gov/ks/nwis/wys_rpt/?site_no=06890900) | +1.571 可含尺度贡献；未输入的调度/下游水位可能解释剩余困难或 H–Q 关系变化 | “所有负 Q 都是错误”“无 STGQ 所以直接测 Q”“只凭水位就能恢复所有出流” |

### Inferences

- **给协调者的建议表述：**“三站存在不同的困难来源。02322800 对流量尺度诊断条件高度敏感；06890900 的水库调节、回水和特殊计算法有直接 USGS 记录；05422600 有城乡混合及人工排水背景，但未找到该测站下游顶托的明确证据。这些过程支持相应机制解释，尚未证明它们造成本次误差。”外部事实依据为问题二的官方链接；实验方向仅依据任务给定值。
- **H 监督边界：**受顶托或调度控制，不等于 H 数据无效或 H 监督一定有害。需要证明具体时段内共享模型如何响应，才能判断跨任务帮助或干扰；本次仅指出关系可能不稳定。[06890900 站点证据](https://nwis.waterdata.usgs.gov/ks/nwis/wys_rpt/?site_no=06890900)；[02322800 站点提示](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=02322800)
- **已有输出的轻量诊断建议，不执行新的实验：**若协调者已有预测文件，可按每站、每模型种子、同一测试 mask 汇总 r、α、b、μ/σ 和有效 n，并看峰值时差；r 高但 b² 或尺度条件偏差大才支持“形状尚可、均值/尺度为主”。若结果是跨种子平均 NSE，应先逐运行分解再聚合，不能把平均 NSE 当成单条序列直接反算。[诊断依据：原始 NSE 分解](https://ag.purdue.edu/hydrology/wiki/environmental-informatics/modeling-resources/attachments/gupta-et-al-2009.pdf)

### Gaps

- 未获得上述分解结果，因而本笔记不给“尺度误差占比”“相关系数”“H 监督导致负迁移”等未计算数字或定论。
- 使用真实留出流域训练 Q 的 observed 条件只能作为诊断，不能作为严格无 Q 的部署效果。若以后使用测试 Q 做去均值/仿射校正，也只能标为诊断上界，不能替代当前正式协议；本次未进行此类处理。
- 所有模型机制解释均需用现有输出或独立分析核验；本次不扩展训练、不下载全量序列、不修复 pipeline，也不把区域文献值冒充实际 static 属性。
