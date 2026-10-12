# 三个新增困难流域的 USGS 站点事实与支持机制

调查日期：2026-10-10。范围：仅调查 06843500、07154500、07230500 的外部公开资料；不训练、不准备全量数据、不修改数据、模型或实验协议。

证据标签：**本站事实**表示官方资料明确点名本站；**地区事实**表示相应地区或上游工程的官方资料，不自动代表整个站点流域；**实验参照**来自本次任务提供的另一研究者核实结果，本笔记未读取本地指标或静态属性；**支持机制**表示与现象一致的待检验解释，不能写成已证明的模型误差原因；**待验证**表示资料或分析尚不足。

任务边界与实验参照：原 86 流域多在美国东北；14 个候选中 01100561 与原 86 重叠，因此 13 个真正新增，总数 99。fixed holdout 在训练和验证阶段屏蔽候选流域 Q，保留 H。模型输入只有降雨、温度、太阳辐射和静态属性，不含历史 Q、调度、抽水或地下水动态。physical 的 Q 均值与标准差由 area×p_mean 回归预测；observed 使用被留出流域真实训练 Q 的均值与标准差，是诊断条件，不能当作严格 zeroQ 结果。这些均为任务给定条件，本次未重新核验实现。

| 站点 | candidate_fixed dual，三 seed 平均 NSE：physical → observed | 本次调查优先解释的现象 |
| --- | --- | --- |
| 06843500 | −8853.164 → −0.190 | 更换 Q 尺度后灾难性负值消失，优先检验均值／尺度失配 |
| 07154500 | −62.303 → −1.053 | 尺度改善明显，断流和洪峰过程仍可能存在残差 |
| 07230500 | −17.177 → −4.284 | observed 仍很差，均值／尺度解释不足，优先检验水库时序影响 |

表中是实验参照，不是 USGS 统计，也不是本次重算。不能由 NSE 单值推断相关系数、预测偏差方向或训练／测试 Q 方差。若 physical 与 observed 是分别训练的模型，二者差异也不能等同于对同一预测序列做线性校准。

共通解释边界：无当前公开 STGQ 文件不等于直接测量流量，也不等于确认为非 rating。现场流量测验、连续水位观测、连续流量计算是不同层次；本站 measurementmethod 和历史连续 Q 的计算方法仍需明确记录。三站的回水、移动床与 H–Q 非唯一性不能由地理位置或候选标签直接推出。

## 06843500：Republican River at Cambridge, Nebraska

### Takeaway

本站最强证据是总汇水面积与直接产流面积差异很大，以及官方明确记载上游灌溉和水库调控；这支持 area×p_mean 无法完整代表实际 Q 尺度的解释。physical → observed 的巨大改善提示首先核验均值与标准差，observed 仍为负则说明这种诊断条件下仍未达到均值基准。

### Cited Findings

- **本站事实｜身份、坐标、水系、面积。** USGS 正式名称为 Republican River at Cambridge, Nebr.，位于 Nebraska 的 Furnas County；页面 WGS84 坐标为 40.284444、−100.143611。总面积 14,460 mi²，contributing drainage area 为 7,780 mi²；页面当前 HUC8 为 10250009，名称 Harlan County Reservoir。该 HUC 名称本身不能证明本站处于 Harlan 水库回水区。来源：[USGS 站点元数据](https://waterdata.usgs.gov/monitoring-location/USGS-06843500/)。
- **本站事实｜位置关系。** 2003 水年报告印刷页 316 将本站列入 Kansas River Basin 的 Republican River 记录：距 Medicine Creek 汇入口下游 0.5 mi，距 Cambridge diversion dam 上游 1.3 mi。报告给出的历史 HUC 为 10250004，与当前站点页面不同；保留版本差异，不据此猜测站点迁移。来源：[USGS Nebraska 2003 水年报告，页 316–317](https://pubs.usgs.gov/wdr/wdr-ne-03-1/aar2003bookI.pdf#page=27)。
- **本站事实｜取用水与调控。** 同报告明确记载自然流量受上游灌溉开发影响，自 1949 年起受上游水库调控；当年记录评为 good。来源：[USGS Nebraska 2003 水年报告，REMARKS，页 316](https://pubs.usgs.gov/wdr/wdr-ne-03-1/aar2003bookI.pdf#page=27)。
- **本站事实｜历史零流。** 2003 水年统计页记录最低日均流量及年度七日最小值均为 0.00 ft³/s，发生于 2002-10-13；这是历史零流证据，不能推广为本实验全部年份的零流比例。该页长期统计注释以 Harry Strunk Lake（Medicine Creek）开始蓄水为界。来源：[USGS 2003 水年统计，页 317](https://pubs.usgs.gov/wdr/wdr-ne-03-1/aar2003bookI.pdf#page=28)。
- **本站事实｜观测与 rating 线索。** 历史水年报告记载 water-stage recorder；USGS legacy 站点页列出 Rating Curve、USGS Rating Depot 和 Station Manuscript 入口。入口存在不等于已核实当前文件类型，但不支持将本站称为已经确认的非 rating 站点。来源：[USGS 站点原页面](https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=06843500)；[USGS 历史站点说明](https://pubs.usgs.gov/wdr/wdr-ne-03-1/aar2003bookI.pdf#page=27)。
- **地区事实｜地下水联系。** USGS 关于 Republican River Basin 的说明记载流量长期下降，并将 High Plains aquifer 的灌溉开发及地下水—地表水相互作用列为调查对象。这是流域背景，不是本站某个年份流量被抽水减少多少的直接证明。来源：[USGS Fact Sheet 028-99：Republican River Basin Ground-Water Flow Model](https://pubs.usgs.gov/fs/FS-028-99/)。

### Inferences

- **支持机制｜effective area 与产流效率。** 7,780／14,460 约为 0.54，是对官方面积的简单比例，不是本地属性计算。若 physical 回归使用总面积，非直接贡献面积、蒸散及取用水可能使 area×p_mean 对 Q 尺度的跨区域映射失准。约两倍的面积差异本身不能解释 NSE ≈ −8853，不能单独归因；必须同时核对 physical 的实际均值／标准差偏差与测试方差。支持依据：[本站面积](https://waterdata.usgs.gov/monitoring-location/USGS-06843500/)；[本站灌溉与调控说明](https://pubs.usgs.gov/wdr/wdr-ne-03-1/aar2003bookI.pdf#page=27)。
- **支持机制｜尺度误差优先。** 根据任务实验参照，从 −8853.164 改善至 −0.190 与“无 Q 尺度外推严重失准”一致，不能把 −8853 直接理解为洪峰形状全部错误。地下水、灌溉、水库可同时影响平均水量和季节变化；模型没有这些动态输入。上述过程的存在有官方依据，过程与当前误差的因果关系尚未验证。支持依据：[USGS 本站备注](https://pubs.usgs.gov/wdr/wdr-ne-03-1/aar2003bookI.pdf#page=27)；[USGS 流域地下水调查](https://pubs.usgs.gov/fs/FS-028-99/)。
- **支持机制｜observed 的剩余负值。** 提供真实训练 Q 尺度后仍有负 NSE，与缺少调度／抽水动态、训练测试时期流量条件变化、低流量误差或时序偏差均相容；现有证据不能排序这些因素。NSE = −0.190 不证明相关系数低，也不证明真实训练均值／方差适合测试期。
- **不支持的归因。** 不能仅凭 Cambridge diversion dam 位于下游 1.3 mi 就断言本站受其回水；不能把水库 HUC 名称当作回水记录；未取得本站河床砂质、移动控制或 H–Q 环线证据。将“标签质量极差”作为主要原因也缺乏支持，因为所查水年的官方记录等级是 good；该等级仍不能代表所有实验年份。来源：[USGS 本站历史说明](https://pubs.usgs.gov/wdr/wdr-ne-03-1/aar2003bookI.pdf#page=27)。

### Gaps

- **待验证｜最小尺度诊断。** 请负责数值的研究者仅从已有汇总或预测缓存逐 seed 报告 μ_physical／μ_trainQ、σ_physical／σ_trainQ、测试 μ_Q／σ_Q、预测均值偏差／σ_Q、预测标准差比与相关系数；不重新准备输入，不重训。先判断极负 NSE 中偏差项与波动项的占比，再决定是否需要更深调查。
- **待验证｜属性口径。** 本次没有读取本地 static，不能确认 area 使用的是总面积还是 contributing area，不能确认 p_mean 的单位、时段、面积加权口径或当前训练集回归覆盖。这里是按任务分工未读取，并非确认文件不可读；未尝试修 pipeline。
- **待验证｜本站物理条件。** 当前证据未给出测站断面的粒径、移动床、回水、控制断面变化、measurementmethod 或实验期逐年计算方法。建议只查本站历史站点分析与少量现场测验元数据，若没有公开资料就保留未知。
- **待验证｜时间匹配。** 2003 水年证据不能直接证明 2021–2024 的水量机制和数据质量；不以历史零流记录代替当前零流比例。

## 07154500：Cimarron River near Kenton, Oklahoma

### Takeaway

本站具有直接的“上游大量灌溉分流”和“多数年份有断流”证据，不能仅以普通降雨—径流迁移解释。历史 USGS 明确使用过 rating curve；observed 后 NSE 仍为 −1.053，与断流／小流量和少数洪峰的预测残差相容，但未证明具体误差发生在哪些时段。

### Cited Findings

- **本站事实｜身份、坐标、水系、面积。** USGS 正式名称 Cimarron River near Kenton, OK，位于 Cimarron County；当前页面 WGS84 坐标 36.9266990832731、−102.95909880161403，HUC8 11040001（Cimarron Headwaters）。总面积 1,208 mi²，contributing drainage area 1,112 mi²。来源：[USGS 站点元数据](https://waterdata.usgs.gov/monitoring-location/USGS-07154500/)。
- **本站事实｜位置及面积版本差异。** 历史报告将本站列入 Arkansas River Basin，位于 Kenton 东北 1.7 mi，North Carrizo Creek 汇入口上游 1.5 mi、Carrizozo Creek 下游 2.2 mi。1984 水年报告及 2009 统计资料采用总面积 1,106 mi²，其中 68 mi² 可能不贡献径流；与当前 1,208／1,112 mi² 不同。这里报告两个版本，不把旧值当作当前面积，也不以面积差异断言流域发生物理扩张。来源：[USGS New Mexico 1984 水年报告，印刷页 22](https://pubs.usgs.gov/wdr/1984/nm-84-1/report.pdf)；[USGS 2009 流量统计，印刷页 58](https://pubs.usgs.gov/sir/2009/5135/pdf/SIR2009-5135-pages25-159.pdf)。
- **本站事实｜灌溉分流。** 2009 统计资料在本站 REMARKS 明确记载上游大量灌溉分流，并给出约 1967 年开始显著影响流量的判断；这是官方对本站的记载，不是按 Panhandle 区域农业情况猜测。来源：[USGS SIR 2009-5135，本站页 58](https://pubs.usgs.gov/sir/2009/5135/pdf/SIR2009-5135-pages25-159.pdf)。
- **本站事实｜零流、流量记录质量及 rating。** 1984 水年报告明确记载多数年份有时断流；当时 Q < 100 ft³/s 的记录评为 fair，较高流量评为 poor。1965-10-17 的历史最大洪峰由超过 7,000 ft³/s 部分的 rating curve 外延估算，外延依据为 contracted-opening 峰值测量。其性质是历史计算方法说明，不是本站现代 measurementmethod 的结论。来源：[USGS New Mexico 1984 水年报告，印刷页 22](https://pubs.usgs.gov/wdr/1984/nm-84-1/report.pdf)。
- **地区事实｜气候证据需限制空间范围。** Oklahoma DEQ 的 2011 Upper Cimarron TMDL 在表 1-3 将 Kenton 河段的多年平均年雨量列为 17.06 in；该河段／报告建模区不等于 USGS 的全部跨州上游流域。报告相邻正文却称 40.1–42.1 in，与本表冲突，不能静默采信正文。来源：[Oklahoma DEQ Upper Cimarron River TMDL，表 1-3](https://www.deq.ok.gov/wp-content/uploads/water-division/Upper-Cimarron-River-Bacteria_TMDLJuly2011.pdf)。

### Inferences

- **支持机制｜较低净产流与尺度外推。** 上游灌溉分流和历史断流支持“实际净 Q 较气象水量潜势更低”的机制；physical → observed 从 −62.303 改善到 −1.053 与尺度失配一致，但不能据此认定回归一定高估了均值，需看预测偏差符号。支持依据：[USGS 本站灌溉备注](https://pubs.usgs.gov/sir/2009/5135/pdf/SIR2009-5135-pages25-159.pdf)；[USGS 本站零流说明](https://pubs.usgs.gov/wdr/1984/nm-84-1/report.pdf)。
- **支持机制｜断流与少数洪峰。** 间歇流意味着“何时恢复连通、何时有流”可能很难由有限气象窗口和 static 单独确定；少数洪峰的幅值／时刻误差也可能主导平方误差。上述是依据已知断流的解释假说，尚无本实验逐时误差或连通性观测；不能声称测试期方差一定很小。支持依据：[USGS 1984 本站说明及极值](https://pubs.usgs.gov/wdr/1984/nm-84-1/report.pdf)。
- **支持机制｜标签不确定性有历史依据但不能直接移植。** 历史高流量评级 poor 及外延 rating 说明洪水 Q 可有不确定性；需要检查实验期自身的估测／评级记录才能判断其是否重要。历史 1965 洪峰不在任务涉及的现代实验期，不能用它直接解释现代 NSE。支持依据：[USGS 历史流量计算说明](https://pubs.usgs.gov/wdr/1984/nm-84-1/report.pdf)。
- **不支持的归因。** 不能以当前未取得 STGQ 断言直接测流，因为本站历史 rating 使用已有明确反证；不能把整个 Cimarron 下游砂床／盐化／地下水资料套到 Kenton。尚无本站移动床、回水或多值 H–Q 的直接证据。来源：[USGS 本站历史 rating 说明](https://pubs.usgs.gov/wdr/1984/nm-84-1/report.pdf)。

### Gaps

- **待验证｜最小误差检查。** 用现有预测缓存，分别查看观测零流／近零流时的预测正偏差，以及最大 5–10 个已有洪峰事件的时刻与幅度误差；并报告相关系数、均值偏差和标准差比。若尺度修正后相关性尚可而偏差仍大，就不能把剩余负 NSE 归结为形状失败。
- **待验证｜measurementmethod。** 本次未获得现代本站逐条现场测验的方法代码，也未确定连续 Q 是否曾改用 index velocity／其他方法。历史 rating 与定期现场测验可并存，两者不能混为排他分类。
- **待验证｜地形与地质。** Kenton 附近地形和基岩可作为区域背景，但尚未完成对本站控制断面、冲积层和地下水交换的直接核验。气候表中的 17.06 in 不替代本地 p_mean；该报告空间口径与站点流域面积不一致。
- **待验证｜控制因素。** 未找到本站明显大型水库调控、城镇污水主导、回水或床面迁移的站点级证据；这些不列作主要解释。

## 07230500：Little River near Tecumseh, Oklahoma

### Takeaway

本站有最明确的上游供水／防洪水库证据，且其气候背景并非 Kenton 那样少雨；不应把三站统一解释成干旱砂床河流。observed 后仍为 −4.284，支持进一步检验水库蓄放水与时序预测脱耦，同时排除预测均值偏差、测试期尺度漂移和数据质量问题。

### Cited Findings

- **本站事实｜身份、坐标、水系、面积。** USGS 正式名称 Little River near Tecumseh, OK，位于 Pottawatomie County；WGS84 坐标 35.172574156657、−96.9319662295343；HUC8 为 11090203（Little）。总面积及 contributing drainage area 都是 463 mi²。来源：[USGS 站点元数据](https://waterdata.usgs.gov/monitoring-location/USGS-07230500/)。
- **本站事实｜位置与水库影响。** 1982 水年报告将本站列入 Arkansas River Basin，位于 US Highway 177 桥，Dance Creek 下游 1.5 mi、Tecumseh 南 5.0 mi；明确记载自 1965 年起受 Lake Thunderbird 调控或分流，该湖位于本站上游 19.2 mi。报告历史面积为 456 mi²，与当前 463 mi² 分列。来源：[USGS Oklahoma 1982 水年报告，印刷页 129](https://pubs.usgs.gov/wdr/1982/ok-82-1/report.pdf)。
- **本站事实｜流量变化与零流。** 该历史报告列出调控前 1944–1964 水年平均 Q 为 149 ft³/s，调控后 1966–1982 为 75.9 ft³/s，并记载若干年份存在无流时段。两时段气候不同，不能将均值差全部因果归给水库，更不能用旧均值替代实验归一化参数。来源：[USGS Oklahoma 1982 水年报告，印刷页 129](https://pubs.usgs.gov/wdr/1982/ok-82-1/report.pdf)。
- **本站事实｜历史观测记录。** 1982 水年报告记载 water-stage recorder，记录评为 fair；1986 水年报告列有多段估测日流量，也重复 Lake Thunderbird 自 1965 年起的影响备注。water-stage recorder 不是连续 Q 直接测量方法的证据。来源：[USGS 1982 站点说明](https://pubs.usgs.gov/wdr/1982/ok-82-1/report.pdf)；[USGS Oklahoma 1986 水年报告，印刷页 135](https://pubs.usgs.gov/wdr/1986/ok-86-1/report.pdf)。
- **上游工程事实｜供水与防洪，不应改写为大型灌溉工程。** Bureau of Reclamation 的 Norman Project 官方说明：Norman Dam 在 Little River 上，Lake Thunderbird 向 Norman、Del City、Midwest City 供市政水，兼有防洪功能，工程不包括灌溉设施或发电；水从湖中抽取，经管道送往城市。该工程说明结合 USGS 的本站上游位置，构成本站受供水调控的直接证据链。来源：[Bureau of Reclamation：Norman Project](https://www.usbr.gov/projects/index.php?id=366)。
- **地区事实｜气候与地形。** 同工程页的历史地区描述给出年降水约 33–35 in，并描述湖周围起伏的橡树林丘陵。这是工程区背景，不是本站全流域现代气候统计；至少不支持把本站自动归为 Kenton 同等级的少雨环境。湖岸砂质也不能证明本站河床为移动砂床。来源：[Bureau of Reclamation：Norman Project](https://www.usbr.gov/projects/index.php?id=366)。
- **地区事实｜地下水。** USGS 2014 研究明确调查包括 Tecumseh 站的中央 Oklahoma 地区；其区域水资源包括河流、阶地／冲积含水层和下伏基岩含水层。此事实支持地下水作为候选过程，但不能证明本站某个洪峰／断流由地下水或抽水驱动。来源：[USGS 原始研究说明：2012–13 年 CPN 地区地表水与地下水](https://www.usgs.gov/publications/stream-water-and-groundwater-quality-and-near-citizen-potawatomi-nation-tribal)。
- **可用于最小后续核验的公开资料。** Reclamation 发布 Lake Thunderbird／Norman Dam 的 Water Operations Monitoring Data 目录；本次只定位目录，未下载或做时间序列对比。来源：[Reclamation RISE，Record 8565](https://data.usbr.gov/catalog/8565)。

### Inferences

- **支持机制｜时序脱耦比单纯干旱更有本站依据。** 水库蓄水、供水抽取和防洪释放能改变入流到本站 Q 的时间及水量关系；模型没有库容、闸门／泄流、城市取水动态，仅知道降雨／温度／辐射和 static，难以在相同气象输入下区分不同工程状态。水库及其用途是事实，它们是否主导 observed 的 −4.284 则尚待验证。支持依据：[USGS 本站水库备注](https://pubs.usgs.gov/wdr/1982/ok-82-1/report.pdf)；[Reclamation 工程运行用途](https://www.usbr.gov/projects/index.php?id=366)。
- **支持机制｜训练尺度已知不等于时变水量已知。** observed 只补充真实训练 Q 均值和标准差，没有提供历史 Q 序列或测试期库容、供水需求和释放时刻。它改善 physical 但残留很差，与水库运行、间歇零流、非平稳性和事件误差相容；仅凭该 NSE 不能判定波动形状完全错误，也不能排除测试期均值偏差。
- **不支持的归因。** 不能把该站主要解释为“半干旱大型灌溉取水”；工程官方明确市政供水目的。不能把水库在上游当作本站回水证明；上游调节改变 Q 的时间过程，与本站断面 H–Q 的水力多值性不是同一个命题。湖岸砂质、区域含水层均不能证明本站移动床或异常 rating。支持依据：[Reclamation 工程用途与地点](https://www.usbr.gov/projects/index.php?id=366)；[USGS 本站位置](https://pubs.usgs.gov/wdr/1982/ok-82-1/report.pdf)。

### Gaps

- **待验证｜最小调度核验。** 只挑现有预测中误差最大的 3–5 个事件及一段断流／近零流时段，与同期 Reclamation 已有库水位、库容或出库量记录作时间对齐；先核对目录实际提供的变量和时间分辨率。若预测随降雨立即上升而 Q 跟随延迟释放，是支持时序脱耦的证据；若相关性可接受但均值偏差巨大，则应先回到尺度诊断。无需运行全量数据准备或更改协议。入口：[RISE 水库运行目录](https://data.usbr.gov/catalog/8565)。
- **待验证｜地下水和污水。** 尚无本站控制期抽水、地下水位或污水排放量的直接证据。未证明城市废水是本站基流主因，不能将邻近 North Canadian River 站点的城市污水结论移植到 Little River。
- **待验证｜H–Q 唯一性与方法。** 尚无本站回水记录、移动床／控制变动、升降水段环线或现代 measurementmethod 代码。建议限于本站少量现场测验元数据和站点分析，不为此启动全部 Q–H 原序列准备。
- **共同最小指标诊断｜数学定义。** 对单站、单 seed、相同有效评价样本，记真实 Q 均值和标准差为 μ_Q、σ_Q，预测为 μ_ŷ、σ_ŷ，相关系数 r，则平方误差的恒等分解给出：NSE = 2rα − α² − β²，其中 α = σ_ŷ／σ_Q，β = (μ_ŷ − μ_Q)／σ_Q，使用同一总体方差约定。它能区分均值偏差 β²、幅度 α 与时序相关性 r；这是数学诊断，不是已完成的实验分析。若 σ_Q 很小，绝对误差会被放大；但不能仅因零流很多就断定测试方差很小，少数洪峰同样可增大方差。三 seed 平均 NSE 不足以反推出这些分量；需要由负责数值的研究者分别从已有缓存提取。physical／observed 若分别训练，还需核对预测是否只差仿射变换，才能把改善定量归为尺度校正。
- **检索局限与停止边界。** 部分旧 USGS PDF 的网页全文阅读入口失败，已能检索到的官方站点页、报告相关段落与工程说明构成上述核心证据；PDF 具体页码只在来源明确给出时标注。旧报告说明历史机制，不能代替实验期测验方法和逐年数据评级。未读取本地实验表、static 或 Q–H 原序列，未进行训练、数值分析或 pipeline 修复。次要地质、砂床与回水属性保留未知，避免为补齐所有属性扩大调查。
