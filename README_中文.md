# Project Leap 2D——IHC 荧光图像综合分析

[English](README.md)

Project Leap 2D 用于在 Apple Silicon Mac 上分析拆分后的单通道免疫组织化学
（IHC）荧光 Z-stack 图像。程序构建 Whole Cell（整细胞）、Soma（胞体）和
Processes（突起）三类感兴趣区域（ROI），由你在 Fiji 中复核。程序从未经改动
的灰度数据测量指定荧光通道，并生成叠加图、分析报告和 Excel 工作簿。

程序位于 [`Project Leap 2D V1.0.1/`](<Project Leap 2D V1.0.1/>)
文件夹中，仓库根目录存放分发工具和说明。

## 首次安装

本版本适用于运行 macOS 11 或更新版本的 Apple Silicon Mac。首次建立依赖环境
需要联网，无需预先安装 Python、Cellpose 或 Fiji。

1. 下载 [Project-Leap-2D-V1.0.1.zip](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.zip)。
2. 双击 ZIP 解压，保持解压后的文件和文件夹结构不变。
3. 打开 `Project Leap 2D V1.0.1 Distribution` 文件夹，双击其中的
   `install_macos.command`。

Terminal 会自动打开并显示安装进度，无需手动输入命令。安装器会检查随附的
程序文件，将工作包复制到 `~/Desktop/Project Leap 2D V1.0.1`，再建立或检查
共享依赖环境。显示 `INSTALLATION COMPLETE` 后，即可使用桌面工作包中的
`Run Analysis.command`。

ZIP 不必保存在“下载”文件夹；在其他可读取文件并正常运行脚本的位置解压也可以。
请将安装器与随附文件保留在一起。如果目标路径已存在，安装器会停止并保留原有内容。

依赖保存在 `~/Applications/Project Leap 2D Support`。详细安装步骤和其他安装位置
等高级选项见 Release ZIP 中的 `INSTALL_中文.md`。

## 日常运行与修复

在已安装的工作包中：

1. 将一批已经独立备份、拆分为单通道的 Z-stack TIFF 放入 `Sample Image`。
2. 双击 `Run Analysis.command`，按提示完成 Fiji 复核；结果写入 `Result`。
3. 程序文件缺失、损坏或启动检查失败时，双击 `Repair.command`。
   它先检查本版本程序文件，必要时恢复；随后检查依赖环境，必要时重建环境。

每次分析开始前，程序会快速检查本地环境，不访问网络。每次修复都需要联网，
从正式 `v1.0.1` Release 获取并验证文件清单
`Project-Leap-2D-V1.0.1.manifest.json`；需要恢复程序文件时，才下载并验证
`Project-Leap-2D-V1.0.1.zip`。如果无法获取或验证所需的发布信息及文件，
修复便会停止，不再检查或重建环境。

修复会恢复同一版本的程序文件，保留 `Sample Image`、`Result` 和
`Analysis Package/Run State` 中的内容。它不会自动升级，也不能修复该版本
本身已有的程序错误。如果修复脚本缺失或无法启动，请从同一 Release 重新下载。

输入要求、分析路径、Fiji 编辑和输出说明见
[中文使用说明](<Project Leap 2D V1.0.1/README/README CN.md>)。
工作包内的 `Manual Command.txt` 提供通过 Terminal 运行分析和修复脚本的命令。

## 输入备份与运行状态

每批输入都应使用工作副本，并保留独立备份。只有 Fiji 完成、全部验证通过且
五个结果文件一起成功保存到 `Result` 后，本次实际使用的 TIFF 才会移入 macOS
Trash。如果安全检查使程序停止、用户取消、发生异常或结果文件保存失败，
输入文件会保留。

程序将运行锁和恢复记录保存在 `Analysis Package/Run State` 中，请不要手动清空
这个文件夹。Git 不跟踪这些运行文件、输入图像和分析结果。

## 仓库克隆、Code ZIP 与 Release ZIP

Git 克隆包含受版本控制的源码和 Git 历史。GitHub 的
**Code → Download ZIP** 以及自动生成的 **Source code** 压缩包是源码快照，
不包含完整安装包，也不包含 Git 未跟踪的空工作目录。
安装时应使用指定版本的 Release ZIP。

如需查看源码或开发，克隆或解压源码后可在仓库根目录运行
`./prepare_workspace.command`。它在工作包内建立 `Sample Image`、
`Result` 和 `Analysis Package/Run State`，保留已有内容，并拒绝符号链接
或类型不正确的目标目录。该命令只准备工作目录，不安装依赖。

## 支持的系统与科学适用范围

V1.0.1 面向运行 macOS 11 或更新版本的 Apple Silicon Mac。
所需通道和物理标定要求见使用说明。GFAP-only 分析支持成熟星形胶质细胞：
没有年龄标记或明确标记为 `mature` 时使用成熟配置；识别到 `neonatal`
或互相冲突的年龄标记时停止分析。

通过软件完整性与自动化检查，并不代表分析方法已在不同组织、年龄、染色方案、
疾病模型、显微镜或实验室条件下得到生物学验证。请使用适当的参考样本，
验证分析方法是否适用于你的成像条件。

## 许可证与第三方组件

本项目原创代码采用 [Apache License 2.0](LICENSE)。第三方软件和模型继续遵循
各自条款；相关许可证、来源、训练数据说明和引用要求见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [`LICENSES/`](LICENSES/)。
