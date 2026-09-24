# Project Leap 2D

## 日常运行

默认工作包目录为 `~/Desktop/Project Leap 2D V1.0.1`。

```text
Project Leap 2D V1.0.1/
├── Sample Image/
├── Result/
├── Analysis Package/
├── README/
├── Run Analysis.command
├── Repair.command
└── Manual Command.txt
```

`Analysis Package` 保存程序文件，其中的 `Run State` 由程序自动管理，
存放运行锁和恢复记录，请勿随意清除。`README` 文件夹内有 `README CN.md` 和
`README EN.md` 两份说明。

1. 将一批拆分后的单通道 Z-stack TIFF 放入工作包的 `Sample Image`。
2. 双击工作包内的 `Run Analysis.command`。macOS 会在 Terminal 中打开它，
   显示运行进度；按程序提示完成 Fiji 复核。

正式输出直接保存到 `Result` 根目录。日常使用不需要手动输入命令；
如需从 Terminal 启动，可使用 `Manual Command.txt` 中与两个入口等价的命令。

普通启动时，程序只读取小型安装状态文件和记录环境要求的配置文件，
并检查 Python、Fiji 和 Cellpose 模型是否仍在记录的位置。这一步不联网、
不导入模型，也不重新计算大文件哈希；所需模型会在分析阶段载入。
这项快速检查通过，并不表示所有程序和依赖文件的完整性都已得到验证。

## 检查与修复

程序文件缺失、损坏或启动检查失败时，双击工作包内的 `Repair.command`。
无需填写参数或手动选择 ZIP，修复会按以下顺序执行：

1. 使用 macOS 自带工具联网取得本版本的发布校验清单，逐一检查程序文件。
   如有缺失或损坏，自动下载并校验同版本发布包，再恢复相应文件。
2. 程序检查和必要的文件恢复成功后，完整检查由本程序管理的 Python、
   锁定版本的依赖、模型和 Fiji/Java，并确认关键依赖能够导入。
   如有必要，修复会重建依赖环境；环境完好时不重复安装。

恢复程序文件时，不会覆盖或清除 `Sample Image`、`Result` 和
`Analysis Package/Run State` 中的内容，也不会自动升级版本。修复只能恢复
同版本中缺失或损坏的文件，无法解决该版本软件本身已有的 bug。
若 `Repair.command` 本身缺失或无法启动，需要重新取得该版本的修复入口。

在线修复需要本版本的 GitHub Release 及其发布资产已发布且可访问。
无法取得或校验发布资料时，修复停在程序检查阶段，不继续检查或重建环境。
日常分析不联网；`Repair.command` 每次运行都需要联网检查程序发布资料。

完整的环境检查会清除受管 Python 标准库在运行时生成的 `__pycache__`，
因此检查过程会修改这些缓存目录。对于已确认由本程序管理的损坏环境，
每次修复最多尝试一次重建：先暂存旧环境，再安装并验证新环境，
检查通过后才替换；失败则尝试恢复旧环境并停止。无法确认归属的目录保持原样。

分析与修复共用由 macOS 内核持有的环境锁；分析正在使用环境时，修复会
停止，不会移动正在使用的 Python、模型或 Fiji。程序与环境检查全部通过后，
修复才报告完成；随后双击 `Run Analysis.command` 开始分析。

## 输入模式

- 每批必须恰好包含一个 DAPI、至少一个 eGFP/GFAP，以及恰好一个
  KCNN1/KCNN2/KCNN3/KCNJ10 测量通道。所有文件必须是拆分后的单通道
  ZYX TIFF，并具有一致的图像尺寸和物理标定。
- 不要把仍同时包含多个通道的原始总图放进 `Sample Image`；同一文件名
  同时含有 DAPI、GFAP、KCNN2 等通道词时，程序无法确定唯一的通道身份，
  会停止分析。
- 只要识别到 eGFP stack，就使用 eGFP 分析路径；即使
  同时有 GFAP，也不会启用 GFAP-only。
- 无论使用 eGFP-only 还是同时使用 eGFP 和 GFAP，程序都先从 Z-stack
  首尾各排除 5 µm，将目标 9.5 µm 按实际 Z-step 换算为最近层数，并在剩余
  深度中设置 5 个等间距位置。每个位置分别运行 6 个 Morphology Baseline、
  6 个 Structural Refinement 和 6 个 Distributional Threshold 候选，
  先在同一位置内选出 1 个最佳 Whole，再比较 5 个位置的最佳结果。
  位置比较根据每个 DAPI 核区域的逐层归一化 P85（第 85 百分位数）曲线
  识别轴向活跃区段；只有一个轴向核与一个 Whole 连通区域双向唯一对应时，
  才计为可分析细胞，并同时评价该核在窗口内的轴向保留比例。
  测量通道的选区权重为 0，不决定 Z 或 Whole。深度不足时终端显示
  `z_step_limit` 并结束；最高总分仍完全相同时显示 `z_selection_ambiguous`
  并结束。
- DAPI/eGFP 文件名没有年龄标记时，eGFP 路径根据形态自动选择成熟或
  新生期配置。只有 eGFP 结构通道时使用 eGFP-only 年龄校准；
  eGFP 和 GFAP 同时存在时保留既有年龄参数。只有在最终 Z 和初始 Whole
  选定后，程序才按年龄采用相应配置。eGFP-only 参数由两份已知 Mature 和
  三份 P3 Neonatal 参考样本定向校准；其他年龄、脑区或采集条件仍需用
  已知年龄样本验证。明确且不冲突的文件名年龄标记优先。
- 只有 DAPI + GFAP、没有识别到 eGFP 时，自动启用独立 GFAP-only
  路径。包内已附带经过校验和固定的 InstanSeg CPU TorchScript 模型，
  无需联网或安装完整 InstanSeg。它只生成 DAPI 核候选；三维核连接、
  GFAP 关联、排他归属及 Whole/Soma/Processes 仍由本程序验证与构建。
- 测量通道只接受不改变强度的技术可测性检查，五位置评分权重为 0；
  它不参与 Z、Whole 或 ROI 定义，原始灰度仅在最终最佳位置中用于测量。
- 本版本的 GFAP-only 分析仅支持成熟星形胶质细胞。DAPI/GFAP 文件名
  没有年龄标记或明确包含 `mature` 时，程序使用成熟配置；识别到
  `neonatal` 时会在分析前停止，年龄标记互相冲突时也会因输入含糊而停止。

## Fiji 复核与 Cell Edit

- Whole Cell ROI Manager：Delete、Split。
- Soma ROI Manager：Delete、Merge、Enlarge。
- Processes ROI Manager：Delete、Trim Processes。
- Revert 使用后进先出撤销栈；可连续点击，逐步撤销已经提交的 Delete、
  Merge、Split、Enlarge 或 Trim Processes。

Trim Processes 会打开一个允许同时操作图像窗口的非模态面板，
提供 Preview Trim、Clear Cell Draft、Undo、Confirm Trim 和 Cancel，
无须预先选中细胞。可在三个复合图窗口中
任意一个使用自由手绘面积选区；点击 Preview Trim 前，草稿只显示受影响
的细胞 ID 和笔画轮廓。首次预览并行核验全部受影响细胞，以黄色显示预计
删除的 Processes 区域，黑色轮廓仅用于增强对比。正式 Whole/Soma/Processes
轮廓在确认前保持原样。首次预览后，草稿变化会自动重新核验受影响细胞。

Clear Cell Draft 只清除所选细胞 ID 的草稿；即使一笔跨越多个细胞，也只
清除该细胞对应的部分。面板内 Undo 可撤销笔画和清除操作。预览已过期、
没有 Processes 像素可删除、修剪后 Processes 为空或保留分支与其 Soma
断连时，均不能 Confirm Trim。确认后从 Whole 和 Processes 同步删除相同
像素，Soma 保持不变，整次确认作为一步加入全局 Revert 栈；
Cancel 放弃草稿。Trim 不会对后续 Split 或 Enlarge 增加排除约束，
也不写入运行历史文件。

Split 每次只在所选 Whole Cell 及其紧邻局部区域寻找一个额外 DAPI 核，
再基于两个核把一个细胞重新计算为两个。原 Whole 是可信主区域；旧 Whole
外只恢复与对应 Soma 连续、结构证据明确且没有进入其他核竞争区域的
Processes，避免产生宽泛的外圈。未通过自动核验或异常小的第二核必须先经
局部 DAPI 模型确认，ROI 外的边缘弱核不能直接触发拆分。Enlarge 对 eGFP
样本保留原有的局部证据规则；在 GFAP-only 样本中，主要依据完整 DAPI 核
和物理标定的核周范围，GFAP 只辅助外层边界和邻细胞排除。通过验证的新 Soma 可以超出旧 Whole，
新增区域会同步加入 Whole，随后重新计算 `Processes = Whole − Soma`。
所有操作都会同步更新三类 ROI，并重新连续编号。局部计算在独立且受限的进程
中运行，支持超时和取消；证据不足时会用简短英文说明操作被拒绝，不强行产生结果。

## 工作区规则

- `Sample Image` 和 `Result` 是长期保留的固定文件夹。
- `Analysis Package/Run State` 是可见的运行状态文件夹，保存运行锁、恢复信息和
  Matplotlib 缓存；程序不会创建隐藏的 `.runtime` 文件夹。
- 新一轮开始时，如果 `Result` 根目录仍有旧文件，程序会把它们整体移入
  `Pending`、`Pending 1`、`Pending 2`……
- 程序不建立跨运行的图像历史缓存。
- 只有 Fiji 完成、全部验证通过且正式结果成功保存后，本轮实际使用的
  TIFF 才会移入 macOS Trash。
- 保存五个正式输出文件和将源 TIFF 移入 Trash 时，程序都会维护小型恢复记录，
  并以覆盖方式更新。程序被强制终止或电脑断电后，下次启动会先恢复到一致
  状态；恢复完成后记录自动删除，不会按批次累积。
- 安全停止、Fiji Cancel、异常、`Ctrl-C`、`--skip-fiji` 或正式结果保存失败时，
  原图都会保留；未被通道识别器采用的额外文件不会被移动。

## 正式输出

- `IHC_2D_Whole_Astrocyte_Overlay.png`
- `IHC_2D_Astrocyte_Soma_Overlay.png`
- `IHC_2D_Astrocyte_Processes_Overlay.png`
- `IHC_2D_Analysis_Report.txt`
- `IHC_2D_Fluorescence_Results.xlsx`

程序会验证三类 ROI 的连续编号、精确分区、Fiji 原始灰度测量、overlay
尺寸和工作簿结构。全部验证通过后，才将五个文件整组保存，并提供回滚保护。
分析报告记录各主要阶段的整体状态和推理耗时，并列出 5 个 Z 位置的
实际范围与宽度、每位置最佳候选、DAPI 轴向核对应与保留结果、五项分数、
总分和最终最佳位置。
候选清单同时标记每个候选所属位置及是否为该位置的最佳候选，不额外
生成逐候选调试报告。
