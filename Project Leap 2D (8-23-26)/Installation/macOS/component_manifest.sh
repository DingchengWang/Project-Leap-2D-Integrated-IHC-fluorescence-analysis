#!/bin/sh

# Audited external component contract for the macOS installer.
# This file contains data only and is sourced by bootstrap_macos.sh.

PROJECT_LEAP_INSTALL_SCHEMA="2"
PROJECT_LEAP_PLATFORM="macos-arm64"
PROJECT_LEAP_MINIMUM_MACOS_MAJOR="11"
PROJECT_LEAP_PYTHON_VERSION="3.9.25"

PROJECT_LEAP_UV_VERSION="0.11.16"
PROJECT_LEAP_UV_URL="https://github.com/astral-sh/uv/releases/download/0.11.16/uv-aarch64-apple-darwin.tar.gz"
PROJECT_LEAP_UV_SHA256="2b25be1af546be330b340b0a76b99f989daa6d92678fdffb87438e661e9d88fb"
PROJECT_LEAP_UV_ARCHIVE_ROOT="uv-aarch64-apple-darwin"

PROJECT_LEAP_CELLPOSE_MODEL_URL="https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2"
PROJECT_LEAP_CELLPOSE_MODEL_SHA256="0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"

# Fixed Fiji archive published under the official dated ImageJ archive.
PROJECT_LEAP_FIJI_URL="https://downloads.imagej.net/fiji/archive/latest/20260718-0417/fiji-latest-macos-arm64-jdk.zip"
PROJECT_LEAP_FIJI_SHA256="e66a395160b5affc0c2328accb4782918703918c4b7391a79cfc7300299fea72"
PROJECT_LEAP_FIJI_ARCHIVE_KIND="zip"
PROJECT_LEAP_FIJI_LAUNCHER_RELATIVE="Fiji/Fiji.app/Contents/MacOS/fiji-macos-arm64"
