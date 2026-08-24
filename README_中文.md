# Project Leap 2D——IHC 荧光图像综合分析

[English](README.md)

Project Leap 2D 用于分析拆分后的单通道免疫组织化学荧光 Z-stack 图像。
程序在 macOS 上构建并复核 Whole Cell、Soma 和 Processes 三类感兴趣区域，
从未经改动的灰度数据测量指定荧光通道，最后生成经过检查的叠加图、分析报告
和 Excel 工作簿。

可运行的完整程序包位于
[`Project Leap 2D (8-23-26)/`](<Project Leap 2D (8-23-26)/>)。仓库根目录中的
其他文件用于准备和说明 GitHub 发布版本，不包含另一套分析实现。

## 克隆后先准备工作目录

克隆仓库后，在仓库根目录打开 Terminal，运行：

```bash
./prepare_workspace.command
```

该命令会建立程序运行所需的空工作目录。随后进入内层程序包并按照其中的说明
安装和运行：

```bash
cd "Project Leap 2D (8-23-26)"
./Installation/macOS/install_macos.command
```

输入要求、安装检查、Fiji 复核、输出文件和日常启动方法详见内层
[中文使用说明](<Project Leap 2D (8-23-26)/README_中文.md>)。

## 保护原始图像

每批输入图像必须使用已经独立备份的工作副本。Fiji 完成、全部检查通过且
正式结果成功生成后，本次分析实际使用的 TIFF 会移入 macOS Trash。程序在
安全停止、取消、异常或结果生成失败时会保留输入文件；这一保护机制不能替代
独立备份。

## 仓库克隆、自动源码压缩包与 Release ZIP

Git 仓库克隆包含受版本控制的文件和 Git 历史。Git 不保存空目录，因此新克隆
必须先运行 `./prepare_workspace.command`，随后才能使用程序。

GitHub 自动生成的 **Source code** 压缩包只是受版本控制文件的通用快照，不能
保留完整、已准备好的程序包目录结构。需要解压后即可使用的完整程序包时，应
下载单独发布并经过验证的 **Release ZIP**。GitHub 自动生成的源码压缩包与
Release ZIP 不能视为同一交付文件。

## 支持的系统

本版本专为运行 macOS 的 Apple Silicon Mac 设计，不属于 Windows、Linux 或
Intel Mac 版本。最低系统版本和安装要求以内层使用说明为准。

## 当前验证范围

工程检查覆盖固定程序文件、安装与发布规则、分析过程中必须保持的关系，以及
可重复的测试输出。本版本的 GFAP-only 分析仅支持成熟星形胶质细胞。文件名没有
年龄标记或明确包含 `mature` 时，程序使用成熟 GFAP-only 配置；识别到
`neonatal` 时，程序会在分析前停止。作者已使用仓库外的成熟 GFAP-only 样本
测试程序，这些样本及其结果不随仓库发布；自动化测试也包含合成 GFAP-only
用例。当前验证范围不能证明程序已在不同组织、发育阶段、染色方案、疾病模型、
显微镜或实验室条件下得到广泛生物学验证。

## 许可证与第三方组件

本项目原创代码采用 [Apache License 2.0](LICENSE)。第三方软件、模型、训练
数据说明、许可证、来源和应引用的论文继续遵循各自的条款，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [`LICENSES/`](LICENSES/)。
