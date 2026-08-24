# Project Leap 2D

## 第一次使用

本版本面向 Apple Silicon Mac（macOS 11 或更新版本）。新用户不需要预先
安装 Homebrew、pip、Python、Cellpose 或 Fiji。在 Terminal 中打开
`Project Leap 2D (8-23-26)` 程序包目录，然后运行：

```bash
./Installation/macOS/install_macos.command
```

安装器会把经过固定版本和 SHA-256 校验的 Python 3.9、科学依赖、
Cellpose 模型和 Fiji 安装到可见目录
`~/Applications/Project Leap 2D Support`。安装过程不需要管理员权限；
本版本的空白安装实测联网下载约 2.1 GB，安装后约占 2.8 GB
（文件系统统计可能略有差异）。依赖组合没有改变时，
以后更新 Project Leap 2D 不会重复安装整套环境。

日常启动只读取一个小型安装状态文件，并检查 Python、Fiji 和 Cellpose
模型是否仍在原位置；不会联网、不会导入模型，也不会重新计算大文件哈希。
用户主动运行无参数安装命令时，安装器会先做一次不联网的深度完整性检查，
包括受管 Python 解释器、运行库与标准库，39 个锁定 Python wheel 的约
1.9 万个登记文件、Cellpose 与内置 InstanSeg 模型哈希、Fiji/Java 固定
文件树和关键 Python 导入。完整性基线自身也使用固定 SHA-256 校验。Python
标准库运行时生成的 `__pycache__` 会在比较前清除，深检使用 `-B` 防止
再次生成；wheel 正式登记的预编译 `.pyc` 会保留并逐文件验证。Fiji 正常
运行会更新的 `.checksums` 与 `db.xml.gz` 不参与固定树比较；Python
可执行文件和运行库的 macOS 代码签名仍会单独验证。19 个由 uv 生成的
命令入口只规范化其中随用户名变化的虚拟环境绝对路径，其余内容仍参与
哈希，因此可在不同 Mac 用户路径下使用而不会误报损坏。安装入口还会先验证
安装脚本、检查器、契约和三份完整性基线，损坏时在联网或改动安装目录前
停止。环境完好
时直接退出，不下载也不重复安装。确认当前环境由本程序拥有且已经损坏时，
安装器只进行一次自动修复：先暂存旧环境，再安装并验证全新环境；新环境
通过全部检查后才替换旧环境，失败则恢复旧环境并停止。无法确认所有权的
目录不会被删除。

如只需快速确认安装状态和关键路径，可运行：

```bash
./Installation/macOS/install_macos.command --check
```

`INSTALLING` 会记录当前环境契约和准确 release 路径，表示上一次安装在
完成验证前被中断。身份匹配时，安装器不会续接这个无法证明完整的环境，
而会清除该未完成目录并从头安装；身份不明的旧标记或目录保持原样并安全
停止。若中断发生在自动修复期间，下一次运行会先恢复旧环境或确认新环境
已完整提交，再继续处理。分析与安装共用一个由 macOS 内核持有的环境锁；
进程退出、异常或断电时锁会由系统自动释放，不需要删除残留 PID 记录。
分析正在运行时，安装或修复会安全停止，不会移动正在使用的 Python、模型
或 Fiji。

## 运行方法

1. 将一批拆分后的单通道 Z-stack TIFF 放入 `Original Image`。
2. 在 Terminal 中打开 `Project Leap 2D (8-23-26)` 程序包目录，
   然后运行：

```bash
./run_project_leap_2d.command
```

`RUN_COMMAND.txt` 中也保存了这些相对命令。正式输出直接写入
`Result` 根目录。

## 输入模式

- 每批必须恰好包含一个 DAPI、至少一个 eGFP/GFAP，以及恰好一个
  KCNN1/KCNN2/KCNN3/KCNJ10 测量通道。所有文件必须是拆分后的单通道
  ZYX TIFF，并具有一致的图像尺寸和物理标定。
- 不要把仍同时包含多个通道的原始总图放进 `Original Image`；同一文件名
  同时含有 DAPI、GFAP、KCNN2 等通道词时，程序会因通道身份不唯一而安全
  停止。
- 只要识别到 eGFP stack，就使用经过验证的 eGFP 分析路径；即使
  同时有 GFAP，也不会启用 GFAP-only。
- DAPI/eGFP 文件名没有年龄标记时，eGFP 路径继续根据形态自动选择成熟或
  新生期配置；明确且不冲突的年龄标记优先。
- 只有 DAPI + GFAP、没有识别到 eGFP 时，自动启用独立 GFAP-only
  路径。包内已附带经过校验和固定的 InstanSeg CPU TorchScript 模型，
  无需联网或安装完整 InstanSeg。它只生成 DAPI 核候选；三维核连接、
  GFAP 关联、排他归属及 Whole/Soma/Processes 仍由本程序验证与构建。
- 测量通道始终只用于最终原始灰度测量，不参与 ROI 定义。
- 本版本的 GFAP-only 分析仅支持成熟星形胶质细胞。DAPI/GFAP 文件名
  没有年龄标记或明确包含 `mature` 时，程序使用成熟配置；识别到
  `neonatal` 时会在分析前停止，年龄标记互相冲突时也会因输入含糊而停止。
- 作者已使用程序包外的成熟 GFAP-only 样本测试程序，这些样本及其
  结果不随程序包发布。

## Fiji 复核与 Cell Edit

- Whole Cell ROI Manager：Delete、Split。
- Soma ROI Manager：Delete、Merge、Enlarge。
- Processes ROI Manager：Delete。
- Revert 使用后进先出撤销栈；可连续点击，逐步撤销已经提交的 Delete、
  Merge、Split 或 Enlarge。

Split 每次只在所选 Whole Cell 及其紧邻局部区域寻找一个额外 DAPI 核，
再基于两个核把一个细胞重新计算为两个。原 Whole 是可信主区域；旧 Whole
外只恢复与对应 Soma 连续、结构证据明确且没有进入其他核竞争区域的
Processes，避免产生宽泛的外圈。未通过自动核验或异常小的第二核必须先经
局部 DAPI 模型确认，ROI 外的边缘弱核不能直接触发拆分。Enlarge 对 eGFP
样本保留原有局部证据
规则；在 GFAP-only 样本中主要依据完整 DAPI 核和物理标定的核周范围，
GFAP 只辅助外层边界和邻细胞排除。通过验证的新 Soma 可以超出旧 Whole，
新增区域会同步加入 Whole，随后重新计算 `Processes = Whole − Soma`。
所有操作都会同步更新三类 ROI，并重新连续编号。局部计算在独立受限进程
中运行，支持超时和取消；证据不足时会用简短英文拒绝，不强行产生结果。

## 工作区规则

- `Original Image` 和 `Result` 是长期保留的固定文件夹。
- `Runtime` 是可见的运行状态文件夹，保存运行锁、恢复信息和
  Matplotlib 缓存；程序不会创建隐藏的 `.runtime` 文件夹。
- 新一轮开始时，如果 `Result` 根目录仍有旧文件，程序会把它们整体移入
  `Pending`、`Pending 1`、`Pending 2`……
- 程序不建立跨运行的图像历史缓存。
- 只有 Fiji 完成、全部验证通过且正式结果成功发布后，本轮实际使用的
  TIFF 才会移入 macOS Trash。
- 正式五文件发布和源 TIFF 移入 Trash 都带有小型覆盖式恢复记录。程序被
  强制终止或电脑断电后，下次启动会先恢复到一致状态；恢复完成后记录自动
  删除，不会按批次累积。
- 安全停止、Fiji Cancel、异常、`Ctrl-C`、`--skip-fiji` 或发布失败时，
  原图都会保留；未被通道识别器采用的额外文件不会被移动。

## 正式输出

- `IHC_2D_Whole_Astrocyte_Overlay.png`
- `IHC_2D_Astrocyte_Soma_Overlay.png`
- `IHC_2D_Astrocyte_Processes_Overlay.png`
- `IHC_2D_Analysis_Report.txt`
- `IHC_2D_Fluorescence_Results.xlsx`

程序会验证三类 ROI 的连续编号、精确分区、Fiji 原始灰度测量、overlay
尺寸和工作簿结构，验证成功后才以带回滚保护的方式整组发布五个文件。
分析报告保留各大阶段的整体状态和推理耗时，不额外生成逐候选调试报告。

完整模块职责见 [MODULE_MAP_中文.md](MODULE_MAP_中文.md)。
`fallback/single_file_fallback.py` 仅用于 eGFP 应急处理和内部一致性检查；
GFAP-only 分析必须使用标准启动器。该 fallback 不是日常入口。
