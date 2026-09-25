# Install Project Leap 2D V1.0.1

This release supports Apple Silicon Macs running macOS 11 or later. The first
installation needs internet access to set up Python, Cellpose, Fiji, and the
other dependencies. You do not need to preinstall Homebrew, pip, or any of these
programs, and administrator access is not required.

## Download and install

1. Download [Project-Leap-2D-V1.0.1.zip](https://github.com/DingchengWang/Project-Leap-2D-Integrated-IHC-fluorescence-analysis/releases/download/v1.0.1/Project-Leap-2D-V1.0.1.zip).
2. Double-click the ZIP to extract it. Keep the extracted files and folders in
   their original layout.
3. Open `Project Leap 2D V1.0.1 Distribution` and double-click
   `install_macos.command`.

Terminal opens automatically to show installation progress; you do not need to
type any commands. The installer verifies the bundled working package, copies
it to `~/Desktop/Project Leap 2D V1.0.1`, and then calls the package's maintenance
programs to create or check the dependency environment. When it reports
`INSTALLATION COMPLETE`, use `Run Analysis.command` in the Desktop package.

The ZIP does not need to be in Downloads. You can extract it in another location
where the files are readable and scripts can run. Keep the installer and its
accompanying files together. Installation stops if the target path already
exists as a folder, file, or symbolic link, so existing inputs, results, and
working packages are not overwritten.

GitHub's automatically generated **Source code** archives provide the source
files. Use the prepared ZIP above for installation.

Python 3.9, the scientific computing dependencies, the Cellpose model, and Fiji
use fixed versions verified with SHA-256. They are installed in
`~/Applications/Project Leap 2D Support`. Later working packages reuse this
environment if they require the same dependency versions and configuration and
the existing environment passes a full check. In that case, no download or
reinstallation is needed.

## Advanced options

To run the installer from Terminal, open the extracted outer folder in Terminal
and run:

```bash
./install_macos.command
```

To install the working package elsewhere, specify an absolute path that does
not yet exist. Its parent directory must already exist and be writable; the
installer does not create missing parent directories:

```bash
./install_macos.command --destination "/absolute/path/Project Leap 2D V1.0.1"
```

To check whether the source working package, checksum manifest, and installation
target meet the requirements, run:

```bash
./install_macos.command --dry-run
```

`--dry-run` does not copy the working package, access the network, or check or
repair the dependency environment. Passing this check does not establish that
the dependencies are installed or that analysis can run.

## After installation

Open the installed working package, place one image batch in `Sample Image`,
and double-click `Run Analysis.command`. The program checks the environment
briefly at startup; daily analysis does not access the network. See
`README/README EN.md` in the package for input requirements, Fiji review steps,
and outputs. To start either entry from Terminal, use the corresponding command
in `Manual Command.txt`.

If program files are missing or damaged, or startup checks fail, double-click
`Repair.command`. It first uses macOS system tools to check this version's
program files online. When files need to be restored, it automatically downloads
and verifies the release package for the same version. After program files have
been checked and any restoration succeeds, it fully checks the dependency
environment and rebuilds it if needed. You do not need to select a ZIP, and
repair does not upgrade the version.

Program-file restoration does not overwrite or clear the contents of
`Sample Image`, `Result`, or `Analysis Package/Run State`. Repair restores the
files that belong to this version; it cannot fix bugs already present in the
version itself. If `Repair.command` is missing or cannot start, obtain the
repair entry for this version again.

Online repair needs access to the GitHub `v1.0.1` Release and its ZIP and JSON
checksum manifest. If these release files cannot be retrieved or verified,
repair stops without checking or rebuilding the dependency environment. See
“Check and repair” in `README/README EN.md` for the full rules.

If the working package has been copied to its destination but environment setup
fails, the installer reports the error and leaves the package in place.
Double-click `Repair.command` inside it to check and repair the installation;
do not delete existing inputs or results to reinstall.

After installation, daily use needs only the installed working package and the
shared dependency environment. Once you have confirmed that the package works,
you may delete the downloaded distribution; the installer does not delete it
automatically.
