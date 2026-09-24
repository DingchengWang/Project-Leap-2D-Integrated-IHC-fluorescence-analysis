# 安装 Project Leap 2D V1.0.1

本版本适用于运行 macOS 11 或更新版本的 Apple Silicon Mac。首次安装需要联网，
安装器会准备 Python、Cellpose、Fiji 和其他依赖。无需预先安装 Homebrew、pip 或
这些软件，也不需要管理员权限。

## 下载与校验

从同一个 `v1.0.1` Release 下载 `Project-Leap-2D-V1.0.1.zip` 和
`Project-Leap-2D-V1.0.1.zip.sha256`，在这两个文件所在的目录运行：

```bash
shasum -a 256 -c Project-Leap-2D-V1.0.1.zip.sha256
```

看到 `OK` 后再解压 ZIP。`.zip.sha256` 用于手动核对下载文件是否完整。
`Repair.command` 另用 JSON 校验清单和 GitHub 提供的文件摘要校验下载内容，
不读取这个 `.zip.sha256` 文件。

## 首次安装

解压后，将 `install_macos.command`、`payload_sha256.txt` 和 `Project Leap 2D V1.0.1`
文件夹放在一起，双击外层的 `install_macos.command`。macOS 会在 Terminal
中打开安装器。

也可以在 Terminal 中进入解压后的外层目录，运行：

```bash
./install_macos.command
```

安装器先校验随附的工作包，再将它复制到 `~/Desktop/Project Leap 2D V1.0.1`，
随后调用包内的维护程序建立或检查依赖环境。如果目标路径已存在，无论是文件夹、
文件还是符号链接，安装都会停止，因此已有的输入、结果和工作包不会被覆盖。

Python 3.9、科学计算依赖、Cellpose 模型和 Fiji 均使用固定版本，并经 SHA-256
校验，安装位置是 `~/Applications/Project Leap 2D Support`。以后安装的工作包
如果要求相同的依赖版本和配置，且现有环境通过完整检查，就会复用该环境，
无需重新下载和安装。

如需将工作包安装到其他位置，请指定一个尚不存在的绝对路径。其父目录须已存在
且可写；安装器不会自动补建父目录：

```bash
./install_macos.command --destination "/absolute/path/Project Leap 2D V1.0.1"
```

如果只想检查源工作包、校验清单和安装目标是否符合要求，可以运行：

```bash
./install_macos.command --dry-run
```

`--dry-run` 不复制工作包、不联网，也不检查或修复依赖环境。通过这项检查
并不表示依赖已安装或分析已经可以运行。

## 安装后使用

打开已安装的工作包，将一批图像放入 `Sample Image`，然后双击
`Run Analysis.command`。程序启动时会快速检查环境，日常分析不联网。
输入要求、Fiji 复核步骤和输出说明见包内的 `README/README CN.md`。
需要从 Terminal 启动时，可使用 `Manual Command.txt` 中与两个入口对应的命令。

如果程序文件缺失、损坏或启动检查失败，双击 `Repair.command`。它先使用
macOS 自带工具联网核对本版本的程序文件；需要恢复文件时，会自动下载并校验
同版本发布包。程序文件检查和恢复成功后，它会完整检查依赖环境，并按需重建。
这个过程无需手动选择 ZIP，也不会自动升级版本。

程序文件恢复不会覆盖或清除 `Sample Image`、`Result` 和
`Analysis Package/Run State` 中的内容。修复只能恢复本版本应有的文件，
不能消除本版本软件已有的 bug。如果 `Repair.command` 自身缺失或无法启动，
需要重新取得本版本的修复入口。

在线修复需要访问 GitHub 上的 `v1.0.1` Release 及其 ZIP 和 JSON 校验清单。
如果无法取得或校验这些发布文件，修复会停止，不再检查或重建依赖环境。
完整规则见 `README/README CN.md` 的“检查与修复”。

如果工作包已经复制到目标位置，但依赖环境建立失败，安装器会报错并保留该工作包。
可双击包内的 `Repair.command` 检查并修复；不要为了重新安装而删除已有输入或结果。

安装成功后，日常使用只需要已安装的工作包和共享依赖环境。确认工作包可用后，
可以自行删除下载包；安装器不会自动删除它。
