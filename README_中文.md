# Project Leap 2D——IHC 荧光图像综合分析

[English](README.md) · [Project Leap 网站](https://dingchengwang.github.io/Project-Leap/zh/) · [使用指南](docs/guide.zh.md) · [分析方法](docs/methods.zh.md)

Project Leap 2D 用于测量星形胶质细胞整细胞、胞体和突起区域内的目标通道荧光强度。程序依据 DAPI 及 eGFP 和／或 GFAP 选择分析深度、构建细胞区域；用户在 Fiji 中审核 ROI 后，程序在所选 Z 范围的二维最大强度投影上测量目标通道的原始灰度。

本仓库提供“**星形胶质细胞荧光强度分析**”模块 **v1.1.1** 的源码和文档。请从 [Project Leap 网站](https://dingchengwang.github.io/Project-Leap/zh/)下载 **Project Leap v1.0.0**，再在主程序中安装模块。各版本的变化见[版本记录](CHANGELOG.md)。

## 准备图像与安装模块

使用 **macOS 27 或更新版本的 Apple Silicon Mac**。打开 Project Leap 的“**功能与维护**”，安装“**星形胶质细胞荧光强度分析**”，等待显示“**模块已就绪**”。主程序会一并安装所需的科学计算环境、模型和 Fiji。

准备一个多通道 `.tif` 或 `.tiff`，分别包含 DAPI、eGFP 和／或 GFAP，以及 KCNN1、KCNN2、KCNN3、KCNJ10 中的一个待测标记物。图像应保留 Z 轴及像素、切片的物理间距。显微镜原生采集文件需先导出为带标定的 TIFF，合并后的 RGB 彩色图不适用。结构通道仅使用 GFAP、没有 eGFP 时，该分析分支仅适用于成熟星形胶质细胞。

## 分析与审核

自动分析生成整细胞、胞体和突起三类关联区域，供研究者在 Fiji 中检查和修正。[使用指南](docs/guide.zh.md)说明区域审核、完成测量和查看结果的操作；[分析方法](docs/methods.zh.md)解释区域和测量值的计算依据。

## 查看与保存结果

每次完成的任务保存三幅 PNG 叠加图、一份 TXT 报告、一份 XLSX 工作簿及 `IHC_2D_ROI_Archive.zip`。工作簿分别记录整细胞、胞体和突起的面积与荧光测量值，并保留细胞编号、所选 Z 范围和审核记录。目标通道强度保留源图的灰度尺度，未做背景扣除或强度归一化。

ROI 归档保存最终标签图、ImageJ ROI 及其与测量结果的对应关系，可解压查看或提取。请将结果文件夹与原始 TIFF 一并保留。

## 保留的 v1.0.1 独立脚本包

旧脚本包继续供已有用户使用。其安装方式、输入处理和系统要求以旧版文档为准：

- [v1.0.1 Release](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/tag/v1.0.1)、[安装 ZIP](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.zip)和[修复清单](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.manifest.json)。
- [安装说明](distribution/INSTALL_中文.md)和[旧版使用说明](<Project Leap 2D V1.0.1/README/README CN.md>)。
- [`Project Leap 2D V1.0.1/`](<Project Leap 2D V1.0.1/>) 中保留 `Run Analysis.command`、`Repair.command` 和 `Manual Command.txt`。旧版 `Repair.command` 依赖固定的 v1.0.1 Release 及对应附件。

旧包要求将已备份、预先拆分的输入副本放入 `Sample Image`，成功处理后会将实际使用的输入移入 macOS 废纸篓。使用旧包时请遵循其说明。GitHub 的“**Code → Download ZIP**”提供源码快照，应用安装包通过官网获取。

## 源码与贡献

当前模块代码位于 [`engine/project_leap_2d/`](engine/project_leap_2d/)，模块测试位于 [`tests/`](tests/)；桌面界面和组件管理位于 [Project Leap 主仓库](https://github.com/DingchengWang/Project-Leap)。旧脚本包及其分发工具与当前模块分别保留。

Dingcheng Wang 负责科学判定标准、流程决策和审核要求；AI 辅助代码实现、调试与打包。

本项目原创代码采用 [Apache License 2.0](LICENSE)。第三方软件和模型遵循各自条款；来源、许可证和引用要求见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [`LICENSES/`](LICENSES/)。
