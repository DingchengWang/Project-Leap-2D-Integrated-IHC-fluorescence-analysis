# Project Leap 2D——IHC 荧光图像综合分析

[English](README.md) · [Project Leap 网站](https://dingchengwang.github.io/Project-Leap/zh/) · [使用指南](docs/guide.zh.md) · [分析方法](docs/methods.zh.md)

Project Leap 2D 在星形胶质细胞的整细胞、胞体和突起区域内测量指定通道的荧光强度。**1.1.0** 版作为 Project Leap 的“**星形胶质细胞荧光强度分析**”模块运行：导入多通道 TIFF，按提示选择图像组和时间点，再指定通道角色。软件自动生成单通道工作副本，运行既有分析流程，打开 Fiji 供你审核 ROI，最后保存六个结果文件。原始 TIFF 保持原位置，内容不变。

本仓库提供模块源码和文档。本次提供 v1.1.0 模块，由 Project Leap 主程序负责安装与维护。主程序下载仍在准备中，下载状态见 [Project Leap 网站](https://dingchengwang.github.io/Project-Leap/zh/)。

## 使用模块

当前 Project Leap 主程序面向 **macOS 27 或更新版本的 Apple Silicon Mac**。在“**功能与维护**”中安装“**星形胶质细胞荧光强度分析**”，等待显示“**模块已就绪**”。主程序管理模块、科学计算环境、模型和 Fiji，无需单独手动安装模块包。

1. 导入受支持的 `.tif` 或 `.tiff`，其中包含 DAPI、eGFP 和／或 GFAP，以及 KCNN1、KCNN2、KCNN3、KCNJ10 中的一个待测标记物。
2. 按提示选择图像组和时间点，确认通道映射，设置“**图像年龄**”。
3. 点击“**开始分析**”。软件将所选通道拆分为工作副本，并自动分析。
4. 在 Fiji 中检查拟定的细胞区域，按需修改，再结束审核。
5. 主程序显示“**已完成**”后，点击“**查看分析结果**”，检查叠加图、报告、测量表和最终 ROI 归档。

[使用指南](docs/guide.zh.md)逐步说明控件位置、操作反馈和编辑被拒绝时的处理方法。图像元数据由程序自动检查；通道身份和细胞边界需要你判断。GFAP-only 路径仅支持成熟星形胶质细胞。

## v1.1.0 的变化

| 环节 | v1.1.0 模块 |
|---|---|
| 输入准备 | 直接导入一个多通道 TIFF，在界面中选择图像组、时间点和源通道；软件自动生成单通道工作副本。 |
| 原图保留 | 原始 TIFF 只读且保留原位置；工作副本写入独立任务目录。 |
| 安装与结果管理 | Project Leap 统一管理组件、任务状态、结果文件和历史记录。 |
| 最终 ROI 保留 | 在原有五个结果之外新增 `IHC_2D_ROI_Archive.zip`，保存最终区域形状及其与测量结果的对应关系。 |

科学规则版本仍为 **1.0.1**。Z 范围选择、自动分割、年龄配置选择、包含 Trim Processes 的 Fiji 编辑和原始灰度测量均沿用既有流程。详见[分析方法](docs/methods.zh.md)和[版本记录](CHANGELOG.md)。

## 保留的 v1.0.1 独立脚本包

旧脚本包继续供已有用户使用。其安装方式、输入处理和系统要求以旧版文档为准：

- [v1.0.1 Release](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/tag/v1.0.1)、[安装 ZIP](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.zip)和[修复清单](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.manifest.json)。
- [安装说明](distribution/INSTALL_中文.md)和[旧版使用说明](<Project Leap 2D V1.0.1/README/README CN.md>)。
- [`Project Leap 2D V1.0.1/`](<Project Leap 2D V1.0.1/>) 中保留 `Run Analysis.command`、`Repair.command` 和 `Manual Command.txt`。旧版 `Repair.command` 依赖固定的 v1.0.1 Release 及对应附件。

旧包要求将已备份、预先拆分的输入副本放入 `Sample Image`，成功处理后会将实际使用的输入移入 macOS 废纸篓。使用旧包时请遵循其说明。GitHub 的“**Code → Download ZIP**”提供源码快照，不能当作应用安装器。

## 源码与贡献

当前模块代码位于 [`engine/project_leap_2d/`](engine/project_leap_2d/)，模块测试位于 [`tests/`](tests/)；桌面界面和组件管理位于 [Project Leap 主仓库](https://github.com/DingchengWang/Project-Leap)。旧脚本包及其分发工具与当前模块分别保留。

Dingcheng Wang 负责科学判定标准、流程决策和审核要求；AI 辅助代码实现、调试与打包。

本项目原创代码采用 [Apache License 2.0](LICENSE)。第三方软件和模型遵循各自条款；来源、许可证和引用要求见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [`LICENSES/`](LICENSES/)。
