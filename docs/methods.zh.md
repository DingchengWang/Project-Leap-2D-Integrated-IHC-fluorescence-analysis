# 分析方法与测量范围

[English](methods.en.md) · [项目说明](../README_中文.md) · [使用指南](guide.zh.md)

Project Leap 2D 构建星形胶质细胞的二维区域，并在其中测量一个目标通道。模块 **1.1.0** 增加 Project Leap 内的输入、任务和结果管理，科学规则仍为 **1.0.1**。原始多通道 TIFF 只读；用户所选的图像组、时间点和通道映射决定了传入分析引擎的单通道工作图像。

## 输入数据

桌面导入器接受具有明确 Z、Y、X 轴及可选 C、T 轴的 TIFF，每个 TIFF 页必须是完整的单个标量 YX 平面。支持 `uint8`、`uint16` 和 `float32`，写入时不转换像素值。每个像素必须只有一个标量样本，支持的光度解释为 MINISBLACK 和 PALETTE。合并后的 RGB 图像和显微镜原生采集文件不能直接作为输入。

程序必须能从受支持的 OME、ImageJ／SCIFIO 或 TIFF 标定字段读取 X、Y、Z 三轴的正物理间距。软件自动检查轴、页面、数据类型和标定。多个图像组或时间点需要明确选择，每次运行只处理所选对象。OME 通道名称唯一且精确匹配时可预填角色，仍由用户确认。

分析需要一个 DAPI 通道、eGFP／GFAP 中至少一个通道，以及 KCNN1、KCNN2、KCNN3、KCNJ10 中的一个待测通道。每种角色对应独立源通道。导入器在原图目录之外生成带标定的 ZYX 工作 TIFF，保留所选像素及数据类型；引擎随后检查各通道几何尺寸和标定是否一致。

## 结构通道决定分析路径

DAPI 提供细胞核证据，eGFP 和／或 GFAP 提供细胞结构证据。待测通道提供最终测量的荧光值，不参与自动 ROI 定义或分析深度选择。

| 选定的结构通道 | 分析路径及年龄配置 |
|---|---|
| eGFP，可同时含 GFAP | eGFP 路径比较五个 Z 窗口，每个窗口含 18 个候选配置。“自动”依据形态选择 Mature 或 Neonatal 处理配置，用户明确指定的类别优先。 |
| GFAP，不含 eGFP | 独立的 GFAP-only 路径依据 DAPI 和 GFAP 分析，仅支持成熟星形胶质细胞。“自动”使用成熟配置，Neonatal 会被拒绝。 |

### eGFP 路径：选择深度，再构建胞体与突起

程序从 Z-stack 两端分别排除至少 **5.0 µm**，在剩余深度中设置 **五个**等距窗口。目标窗口宽度为 **9.5 µm**：将 `9.5 / Z 层间距` 四舍五入为层间隔数，恰好半个间隔时向上取整，再包括两端平面。因此实际宽度随采集时的 Z 间距而变化。两端排除范围则向上取整为整层。若容纳不下五个不同的起始位置，程序以 `z_step_limit` 停止。

每个窗口中，Morphology Baseline、Structural Refinement 和 Distributional Threshold 三类各六个配置生成整细胞候选区域。同一窗口的 18 个候选共享投影、Cellpose-SAM 分割和预先计算的结构特征。程序先选出窗口内的最佳候选，再根据细胞核关联与轴向保留、整细胞质量、中心对应及光学证据比较五个位置。可分析细胞要求一个轴向 DAPI 细胞核与一个整细胞连通区域双向唯一对应。轴向核证据来自逐层归一化的 DAPI P85 曲线，P85 指核区域内强度的第 85 百分位数。最终位置最高得分完全并列时，以 `z_selection_ambiguous` 停止。

待测通道接受技术质量检查，但在位置评分中的权重为零，不选择 Z 或整细胞区域。年龄配置在最终窗口和初始整细胞区域确定后应用，用于后续处理，不代表识别出了动物的真实年龄。eGFP-only 年龄校准来自两个已知 Mature 和三个 P3 Neonatal 参考样本；其他年龄或成像条件需要适用的已知年龄参考样本。

### GFAP-only 路径：将细胞核与 GFAP 结构关联

程序依据 GFAP 证据选择连续的活跃 Z 范围，由随附的 InstanSeg CPU TorchScript 模型生成 DAPI 核候选，再跨层连接细胞核、检查它们与 GFAP 结构的关联、确定互斥的细胞归属，构建整细胞、胞体和突起。InstanSeg 负责提供核候选，细胞区域由其后的分析构建和验证。区域定义完成后才载入待测通道投影。这条路径不运行 eGFP 的五窗口、90 候选比较。

## 三类区域与 Fiji 审核

Whole Cell 表示保留的完整细胞区域，Soma 表示其中的胞体，Processes 是整细胞扣除胞体后的部分。三套标签共用细胞身份，并满足：

```text
Whole Cell = Soma ∪ Processes
Soma ∩ Processes = ∅
Processes = Whole Cell − Soma
```

Fiji 将区域分别叠加在结构合成图和目标通道原始灰度投影上。用户检查拟定边界，按需使用联动的 Delete、Merge、Split、Enlarge 或 Trim Processes。编辑会同步更新三类区域。Split 和 Enlarge 依据局部证据计算，可拒绝没有充分依据的修改；Merge 使用 **1.0 µm** 的胞体边界邻近条件；Trim 保留胞体，并检查是否使保留的突起产生新的断连。[使用指南](guide.zh.md)在对应操作处说明控件及限制。

Fiji 初始对话框同时提供 **Review ROIs** 和 **Measure Now**。指南采用 Review ROIs 流程，要求用户在接受测量前检查区域。程序分别记录是否进入编辑审核流程及实际发生的编辑；`Manual_Review_Used` 是流程记录，不能证明边界具有生物学正确性。

## 原始灰度测量与结果保存

所选 Z 范围和投影方法决定哪些源像素组成测量图像，最终 ROI 决定测量该图像的哪一部分。Fiji 测量目标通道的原始灰度投影；合成图颜色和显示对比度不是测量数据。程序不会对待测强度做背景扣除或强度归一化。用于构建区域的结构图像处理与这条测量路径分开。

工作簿的 Whole Cell、Soma 和 Processes 工作表包含面积、均值、中位数、最小值、最大值、积分密度和原始积分密度，并记录通道、投影、Z 边界、年龄配置和审核情况。这些数值描述选定的**二维投影**，不等同于细胞三维体积或自动扣除背景后的表达量。实验的采集设置和取样设计仍决定哪些比较有意义。

保存前，程序检查细胞身份同步、三类区域的分割关系、Fiji 测量记录、叠加图尺寸和工作簿结构。v1.1.0 在原有三幅 PNG、TXT 报告和 XLSX 工作簿之外，新增 `IHC_2D_ROI_Archive.zip`。归档包含最终三类标签 TIFF、ImageJ 原生 ROI、身份对应、标定、所选 Z／投影和运行来源。导出器核对这些内容与已完成的 Fiji 测量是否一致，不重新生成或改变 ROI。六个结果共同进入结果保存事务。

技术一致性检查不能确立方法在不同组织、年龄、染色、显微镜或实验室条件下的生物学有效性；应使用适合目标实验的参考样本，并审核区域边界。v1.1.0 的流程变化不构成准确率或速度提升的证据。

## 源码位置

- [深度窗口构建](../engine/project_leap_2d/images/image_processing.py)、[eGFP 流程](../engine/project_leap_2d/analysis_controller.py)及[候选选择](../engine/project_leap_2d/segmentation/candidate_selection.py)。
- [GFAP-only 流程](../engine/project_leap_2d/analysis_modes/gfap_only/gfap_only_pipeline.py)及[细胞归属与区域构建](../engine/project_leap_2d/analysis_modes/gfap_only/gfap_only_analysis.py)。
- [Fiji 审核与测量](../engine/project_leap_2d/fiji_review/resources/astrocyte_roi_reviewer.groovy)、[工作簿输出](../engine/project_leap_2d/reporting/excel_results.py)、[最终 ROI 归档](../engine/project_leap_2d/final_roi_archive.py)及[桌面适配](../engine/project_leap_2d/desktop_adapter.py)。
- TIFF 导入、通道映射、组件管理和桌面界面位于 [Project Leap 主仓库](https://github.com/DingchengWang/Project-Leap)。第三方来源和许可证见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
