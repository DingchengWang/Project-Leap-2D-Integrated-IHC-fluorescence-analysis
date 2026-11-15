# Changelog / 版本记录

## 1.1.1 — shared runtime compatibility / 共享运行环境兼容修复

The module now uses the shared scientific runtime selected by Project Leap, validates its declared runtime and model dependencies, and records the actual runtime identity and file fingerprints in the run provenance. This fixes recognition of the shared runtime during analysis startup. Analysis algorithms, models, and parameters remain unchanged.

[Project Leap v1.0.0](https://github.com/DingchengWang/Project-Leap/releases/tag/v1.0.0) is publicly available. Install or update **Astrocyte Fluorescence Intensity Measurement** through the application's **Modules & Maintenance** page.

本版修复分析启动时对共享科学运行环境的识别：模块使用 Project Leap 选定的环境，校验声明的运行环境和模型依赖，并在运行记录中保存实际环境身份及文件指纹。分析算法、模型和参数保持不变。

[Project Leap v1.0.0](https://github.com/DingchengWang/Project-Leap/releases/tag/v1.0.0) 已公开下载。请在主程序的“**功能与维护**”中安装或更新“**星形胶质细胞荧光强度分析**”。

## 1.1.0 — Project Leap module / Project Leap 功能模块

This release integrates the fluorescence workflow into Project Leap. It introduced the following changes:

| Area | Change in v1.1.0 |
|---|---|
| Input preparation | Import one multichannel TIFF, explicitly select its image series/time point, and assign channel roles in the application. Single-channel working copies are generated automatically. |
| Original images | Source TIFFs are read-only and remain in place; working copies are written to a separate task directory. |
| Installation and results | Project Leap manages components, task status, result files, and history. The module package is installed through the application. |
| Final ROI retention | `IHC_2D_ROI_Archive.zip` joins the three PNGs, TXT report, and XLSX workbook, preserving final geometry and its correspondence to the measurements. |

Z selection, automatic segmentation, age-profile rules, Fiji editing including Trim Processes, and raw-grayscale measurement were retained from the earlier workflow. [Methods](docs/methods.en.md) describes the scientific rules used by the current module.

本版将荧光分析流程接入 Project Leap，并引入以下变化：

| 环节 | v1.1.0 的变化 |
|---|---|
| 输入准备 | 直接导入一个多通道 TIFF，在界面中明确选择图像组、时间点和通道角色；软件自动生成单通道工作副本。 |
| 原图保留 | 原始 TIFF 只读且保留原位置，工作副本写入独立任务目录。 |
| 安装与结果管理 | Project Leap 管理组件、任务状态、结果文件和历史记录，模块包通过主程序安装。 |
| 最终 ROI 保留 | 在原有三幅 PNG、TXT 报告和 XLSX 工作簿之外新增 `IHC_2D_ROI_Archive.zip`，保存最终区域形状及其与测量结果的对应关系。 |

Z 范围选择、自动分割、年龄配置、包含 Trim Processes 的 Fiji 编辑，以及原始灰度测量，均沿用先前流程。当前模块的科学规则见[分析方法](docs/methods.zh.md)。

## 1.0.1 — preserved standalone package / 保留的独立脚本包

The [v1.0.1 Release](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/tag/v1.0.1), its fixed tag, ZIP, and checksum manifest remain available for existing installations and same-version repair. Its source, installer, and usage documentation remain under `Project Leap 2D V1.0.1/`, `distribution/`, and `packaging/`.

保留 v1.0.1 Release、固定标签、ZIP 和校验清单，供已有安装及同版本修复使用。对应源码、安装器和说明继续保存在 `Project Leap 2D V1.0.1/`、`distribution/` 和 `packaging/` 中。
