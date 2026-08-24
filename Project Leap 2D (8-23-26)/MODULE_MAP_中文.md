# Project Leap 2D 模块说明

这份文件说明发布包的目录、模块职责和运行顺序。日常运行只需要执行
`run_project_leap_2d.command`，无需单独启动任何 Python 模块。

## 1. 软件包目录

```text
Project Leap 2D (8-23-26)/
├── run_project_leap_2d.command
├── RUN_COMMAND.txt
├── README_中文.md
├── README_English.md
├── MODULE_MAP_中文.md
├── MODULE_MAP_ENGLISH.md
├── Installation/
│   └── macOS/
├── Original Image/
├── Result/
├── Runtime/
├── project_leap_2d/
├── fallback/
├── tests/
└── validation/
```

- `Original Image/`：放入一批拆分后的单通道 Z-stack TIFF。
- `Result/`：保存本轮五个正式输出；旧结果会先整体移入连续编号的
  `Pending` 文件夹。
- `Runtime/`：可见的运行锁、Cell Edit 事务和临时状态；不属于正式结果。
- `fallback/single_file_fallback.py`：仅用于 eGFP 应急处理和内部一致性检查的
  单文件入口。GFAP-only 必须使用标准启动器；该 fallback 不是日常入口。

## 2. 主入口、路由与共享运行状态

```text
project_leap_2d/
├── __main__.py
├── workspace_launcher.py
├── runtime_loader.py
├── runtime_manifest.py
├── runtime_attributes.py
├── startup.py
├── analysis_workflow.py
├── analysis_controller.py
├── command_line.py
├── settings.py
├── data_structures.py
└── run_state.py
```

- `workspace_launcher.py`：固定使用包内 `Original Image`、`Result` 和
  `Runtime`，处理运行锁、Pending 归档、安全发布和成功后的 Trash 清理。
- `startup.py`：在 NumPy、SciPy、OpenCV 和模型代码之前固定并行线程环境。
- `runtime_loader.py`：按声明顺序把模块装入同一共享运行状态，避免重复
  模型、缓存和全局状态。
- `runtime_manifest.py`：声明装载顺序、必需符号和 Fiji 资源。
- `runtime_attributes.py`：在局部任务中临时替换共享运行状态；无论任务
  正常完成还是发生异常，都会按相反顺序恢复原对象，避免一次操作影响后续
  分析。
- `analysis_workflow.py`：根据通道选择经过验证的 eGFP 主线路或独立 GFAP-only
  线路，并负责测量、Fiji 复核和发布交接。
- `analysis_controller.py`：保持已经验证的 eGFP 科学运行顺序。
- 其余模块提供参数、数据结构、命令行入口、并行状态和运行诊断。

## 3. 图像处理与 eGFP 候选分析

```text
project_leap_2d/images/
├── channel_files.py
├── image_loading.py
└── image_processing.py

project_leap_2d/segmentation/
├── cellpose_segmentation.py
├── candidate_features.py
├── candidate_evaluation.py
├── candidate_selection.py
├── standard_morphology_candidates.py
├── structural_refinement_candidates.py
├── distributional_threshold_candidates.py
└── candidate_catalog.py
```

- `images/`：通道发现、物理标定、ZYX 读取、投影和显示图生成。
- `cellpose_segmentation.py`：在五个 Z 区间运行并复用 Cellpose-SAM。
- `candidate_features.py`：构建可复用的 DAPI、Sato、top-hat 和分布特征。
- 三个候选模块按声明顺序提供 30 个 Morphology Baseline、30 个 Structural
  Refinement 和 30 个 Distributional Threshold 候选，共 90 个。
- 候选生产模块直接写入这些功能名称，报告从候选元数据读取同一名称，
  不经过显示名称翻译层。
- `candidate_evaluation.py` 和 `candidate_selection.py`：并行计算、排序、
  去近重复，并在新候选没有通过明确优势条件时保留既有基线结果。

识别到任何有效 eGFP stack 时都使用这条主线路；GFAP 同时存在也不会
切换到 GFAP-only。测量通道不参与 ROI 定义。

## 4. DAPI、Soma/Processes 与手动重算

```text
project_leap_2d/nuclei/
├── dapi_nuclei.py
├── dapi_3d_inventory.py
├── nucleus_ownership.py
└── instanseg_nucleus_detection.py

project_leap_2d/compartments/
├── cell_separation.py
├── compartment_validation.py
├── soma_completion.py
├── soma_and_processes.py
├── selected_cell_split.py
└── selected_soma_enlargement.py
```

- `dapi_3d_inventory.py`：构建三维 DAPI 核清单，并在异常碎片工作量出现时
  安全停止。
- `command_line.py`提供验证专用的
  `--dapi-fragment-workload-preflight-only`工作量统计，以及用于指定必需
  JSON 目标的`--dapi-fragment-workload-json`。日常分析未显式指定目标时，
  工作量安全停止写入`IHC_2D_DAPI_Fragment_Workload_Failure.json`。
- `nucleus_ownership.py`：在 eGFP 主线路中协调唯一 owner nucleus。
- `instanseg_nucleus_detection.py`：按需加载包内固定的 CPU TorchScript
  模型，只生成 DAPI 核候选；不直接决定 Astrocyte 身份。
- `soma_completion.py` 和 `soma_and_processes.py`：完成核范围补全、同一
  Soma 的小岛连通和最终 `Whole = Soma union Processes` 精确分区。
- `selected_cell_split.py`：每次在选中 Whole 中确认一个额外 DAPI 核，
  围绕两个核重建两个细胞。原 Whole 是可信主区域；外部 Processes 只有在
  与对应 Soma 连续、结构证据明确且没有进入其他核竞争区域时才可恢复。
  位于原 Whole 边缘之外的弱核不能直接触发 Split；未通过自动核验或异常小
  的核必须先经局部 DAPI 模型确认。两个新细胞都必须具有真实 Processes，
  否则整次操作不提交。
- `selected_soma_enlargement.py`：重算选中 Soma。eGFP 样本保留原有局部
  证据规则；GFAP-only 样本主要依据完整 DAPI 核和物理标定的核周范围，
  GFAP 只辅助外边界和邻细胞排除。

Split、Enlarge、Merge、Delete 和 Revert 都必须同步 Whole、Soma 与
Processes，保持连续 ID 和无重叠、无缺口的精确分区。

## 5. 独立 GFAP-only 分析

```text
project_leap_2d/analysis_modes/
├── structural_fluorescence.py
└── gfap_only/
    ├── gfap_only_pipeline.py
    ├── gfap_only_analysis.py
    ├── gfap_structure.py
    ├── gfap_nucleus_ownership.py
    ├── gfap_compartments.py
    └── gfap_post_compartment_quality.py
```

- `structural_fluorescence.py`：eGFP 结构荧光接口。
- `gfap_only_pipeline.py`：GFAP-only 的 Z 选择、DAPI 检测、结构分析、
  compartment 构建、测量/Fiji 交接和内存释放。
- `gfap_only_analysis.py`：组合 GFAP-only 各步骤并执行最终一致性检查；
  在二维核投影发生碰撞时保持已确认 owner 核完整，同时保留邻核的独占区域
  供 Cell Edit 排除使用。
- `gfap_structure.py`：GFAP 背景校正和多尺度纤维结构证据。
- `gfap_nucleus_ownership.py`：连接三维 DAPI 核，所有合格邻核共同参与
  竞争，并结合 Z 位置、结构连续路径和局部 GFAP 关联确定细胞身份。
- `gfap_compartments.py`：以 DAPI 核为 Soma 主体；内部归属计算保留
  `2.10 µm` 搜索范围，最终可见 mature Soma 使用 `1.25 µm` 上限，两者
  相互独立；Processes 沿各细胞的连续结构路径排他分配，对身份不明确的
  结构不强行归属。
- `gfap_post_compartment_quality.py`：发布前剔除核对应或形态明显不可靠的
  GFAP-only 结果。

GFAP-only 仅在存在 DAPI + GFAP 且没有识别到 eGFP 时启用。包内 InstanSeg
模型无需联网或安装完整框架。本版本的 GFAP-only 分析仅支持成熟星形胶质
细胞：文件名没有年龄标记或明确包含 `mature` 时使用成熟配置；识别到
`neonatal` 时在分析前停止，年龄标记互相冲突时也会因输入含糊而停止。作者已使用
程序包外的成熟 GFAP-only 样本测试程序，这些样本及其结果不随程序包发布。

## 6. Fiji 复核与 Cell Edit

```text
project_leap_2d/fiji_review/
├── fiji_launcher.py
├── review_protocol.py
├── cell_editing.py
├── cell_edit_context.py
├── cell_edit_worker.py
├── cell_edit_transactions.py
├── cell_edit_fiji_bridge.py
├── failed_run_retention.py
├── review_validation.py
├── measurement_result_validation.py
└── resources/
    └── astrocyte_roi_reviewer.groovy
```

- `review_protocol.py`：准备三类 label、显示图和 Fiji manifest。
- `cell_edit_context.py`：保存局部重算需要的 DAPI、结构证据、核身份、
  分析模式、物理标定和同步 label。
- `cell_edit_worker.py`：在独立受限进程中执行 Split/Enlarge，支持取消、
  超时、输入哈希和结果验证。
- `cell_edit_transactions.py`：在正式 Split 和 Enlarge 路径中验证候选的
  三类 ROI 状态、维持稳定 Cell UID，并且只在候选状态仍与 Fiji
  当前状态一致时提交。
- `cell_editing.py` 和 `cell_edit_fiji_bridge.py`：把 Fiji 的 Split 和
  Enlarge 请求交给 Python 后台任务。
- `failed_run_retention.py`：Fiji 运行失败时只保留最近一次已确认的失败
  现场，并且只清理受控缓存目录中符合规则的旧失败现场，防止缓存长期累积。
- `review_validation.py`：每次编辑后验证身份、像素变化边界和精确分区。
- `measurement_result_validation.py`：验证最终 Fiji 原始灰度测量、连续 ID、
  面积分区、积分密度和 overlay。
- `astrocyte_roi_reviewer.groovy`：在 Fiji 中直接处理 Delete、Merge 和
  Revert；Split 和 Enlarge 会请求 Python 处理。该脚本同时提供
  Cancel 和三类 ROI Manager。

GFAP-only 进入 Fiji 后继续提供 Merge、Split 和 Enlarge；这些操作调用
GFAP-only 对应的 DAPI/结构证据，不会切换到 eGFP 规则。

## 7. 测量、报告与工作区发布

```text
project_leap_2d/reporting/
├── analysis_report.py
└── excel_results.py

project_leap_2d/workspace/
├── folder_checks.py
├── workspace_preflight.py
├── pending_results.py
├── result_publishing.py
├── publication_recovery.py
├── input_cleanup.py
└── input_cleanup_recovery.py
```

- `analysis_controller.py`读取所选闭区间 Z 范围内未经处理的测量通道灰度，
  生成的投影交给 Fiji 对最终 ROI 测量，绝不参与 ROI 定义。
- `analysis_report.py`：记录输入、通道、物理标定、推理耗时、各大阶段
  整体状态、最终 compartment 与 Fiji 状态。
- 候选与工作量检查的生产模块直接输出功能名称，报告不依赖名称翻译模块。
- `excel_results.py`：生成 Whole、Processes、Soma 的最终测量工作簿。
- `result_publishing.py`：执行已经验证的五个正式文件发布步骤。
- `publication_recovery.py`：发布前保存一份很小且覆盖写入的事务记录，并
  为已有正式文件准备恢复副本。五个文件不是在同一瞬间替换；如果中途断电
  或程序退出，下次启动会回滚到完整旧结果，避免留下新旧混合结果。
- `input_cleanup.py`：只在正式结果验证并发布成功后，把本轮已接受的源 TIFF
  移入 macOS Trash。
- `input_cleanup_recovery.py`：在移动源 TIFF 前记录文件身份和位置；如果
  移动途中中断，下次启动会恢复到 `Original Image` 或确认 Trash 中的完整
  目标，不会静默丢失输入。
- 其余 `workspace/` 模块负责运行前检查和 Pending 归档。正常完成后事务
  记录与恢复副本会清除，不会形成逐次累积的用户报告。

## 8. macOS 首次安装

```text
Installation/macOS/
├── install_macos.command
├── bootstrap_macos.sh
├── environment_installer.sh
├── environment_doctor.sh
├── environment_doctor.py
├── installer_integrity_manifest.sh
├── component_manifest.sh
├── requirements_macos_arm64.lock.txt
├── python_wheel_integrity.json
├── managed_python_integrity.json
├── fiji_tree_integrity.json
└── environment_contract.txt
```

- `install_macos.command`：新 Mac 用户双击或在 Terminal 运行的首次安装
  入口。无参数运行会执行离线深度完整性检查；`--check` 只执行日常级快速
  状态与路径检查。
- `bootstrap_macos.sh` 与 `environment_installer.sh`：在可见的用户支持
  目录中安装固定版本的 Python 环境、依赖、Cellpose 模型和 Fiji，不依赖
  用户预先安装 Homebrew 或 pip。已确认属于当前契约的环境损坏时，只进行
  一次事务式自动修复：旧环境先保留为回滚副本，新环境通过深检后才提交；
  修复失败恢复旧环境，来源不明的目录保持不动。
- `component_manifest.sh`、依赖锁文件和环境契约固定下载来源、版本及
  SHA-256，防止不同电脑安装出不同环境。
- `environment_doctor.sh/.py`：验证受管 Python 解释器、运行库与标准库，
  依赖版本、39 个 Python wheel 的逐文件 `RECORD` 哈希、模型、
  Fiji/Java 固定文件树和包内科学资源。
  `python_wheel_integrity.json` 保护 wheel 自身的 `RECORD`，
  `managed_python_integrity.json` 保护由固定版本 uv 获取的 Python 运行时，
  `fiji_tree_integrity.json` 来自固定 SHA-256 的官方 Fiji ZIP；
  `installer_integrity_manifest.sh` 让 bootstrap 在联网或修改安装目录前
  验证安装器、检查器、契约及三份基线本身。标准库运行时缓存会先清除并用
  `-B` 防止重建，wheel 正式登记的预编译 `.pyc` 则保留并验证；uv 生成
  的 19 个命令入口仅规范化随用户名变化的虚拟环境路径。正常分析
  启动只读取很小的安装状态，不会重复联网、安装或导入整套科学依赖。
- `INSTALLING` 同时记录环境契约和准确路径，只标记尚未验证完成且可证明由
  本安装器创建的环境。再次运行时会删除这个未完成版本并从头安装，不会
  续接未知状态；身份不明的目录保持原样。自动修复中断时会先恢复旧环境，
  或深检并确认新环境已经完整提交。启动器和安装器共用一个由 macOS 内核
  持有的固定环境锁；进程结束或崩溃时系统自动释放，不需要维护或清理 PID
  文件，并可防止修复过程移动正在被分析使用的 Python、模型或 Fiji。

## 9. 总体运行顺序

```text
工作区与通道检查
→ eGFP 优先路由，或独立 GFAP-only 路由
→ Whole/Soma/Processes 构建与验证
→ 测量通道原始灰度投影
→ Fiji 人工复核及可选 Cell Edit
→ 三类 ROI 测量与结果验证
→ 五个正式文件通过可回滚、可恢复的事务发布
→ 已使用源 TIFF 移入 macOS Trash
```

这里的“事务发布”表示最终状态可以恢复为完整的新结果或完整的旧结果；
由于五个独立文件需要依次替换，它不是文件系统层面同一瞬间完成的原子操作。
若发布或 Trash 移动被中断，下一次启动会先恢复，再开始新的分析。

经过验证的 eGFP 执行顺序为：Cellpose-SAM → 90 候选 → 候选排序 → 三维
DAPI 清单 → 核归属 → Soma/Processes。

## 10. 验证与发布门禁

- `validation/source_manifest.json`：科学核心和共享运行时装载顺序。
- `validation/release_contract.json`：运行接口、资源和文档契约。
- `validation/release_package_files.json`：最终发布包文件哈希。
- `tests/`：覆盖模块装载、eGFP 基线、GFAP-only、Split、Enlarge、
  Fiji 事务、测量和发布安全。
- 可选的 InstanSeg 官方参考比较从 `PROJECT_LEAP_INSTANSEG_FIXTURE_DIR`
  读取两个官方 `.npy` 文件所在的绝对目录；测试会校验文件的固定 SHA-256，
  夹具文件不随程序包发布。
- 新版本只有在源码、资源、文档、测试和正式包哈希全部一致后才能交付。
