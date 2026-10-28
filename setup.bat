@echo off
chcp 936 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\vendor;%CD%\.deps"

set "CHECK_ONLY="
set "FORCE="
if /i "%~1"=="--check-only" set "CHECK_ONLY=1"
if /i "%~1"=="--force" set "FORCE=1"

echo ==================================================
echo    柚子社 Galgame 语音合集  --  环境安装 setup.bat
echo ==================================================
echo.
echo 本脚本只把依赖装到当前目录的 vendor 文件夹，不需要管理员权限。
echo.

rem ---------- 1. 找一个带 pip 的 Python ----------
set "PY="
if not defined VOICEPACK_PYTHON goto sp_venv
if not exist "%VOICEPACK_PYTHON%" goto sp_venv
"%VOICEPACK_PYTHON%" -m pip --version >nul 2>nul
if not errorlevel 1 set "PY=%VOICEPACK_PYTHON%"
if defined PY goto sp_found

:sp_venv
if not exist "%CD%\.venv\Scripts\python.exe" goto sp_dsh
"%CD%\.venv\Scripts\python.exe" -m pip --version >nul 2>nul
if not errorlevel 1 set "PY=%CD%\.venv\Scripts\python.exe"
if defined PY goto sp_found

:sp_dsh
if not exist "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" goto sp_path
"%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" -m pip --version >nul 2>nul
if not errorlevel 1 set "PY=%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if defined PY goto sp_found

:sp_path
python -m pip --version >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto sp_found
py -3 -m pip --version >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto sp_found

:sp_found
if not defined PY goto no_python
echo   [OK] Python  : %PY%
"%PY%" -c "import sys; print('       version :', sys.version.split()[0], sys.executable)" 2>nul

rem ---------- 2. 检查 FFmpeg ----------
set "FF=1"
where ffmpeg >nul 2>nul
if errorlevel 1 set "FF=0"
where ffprobe >nul 2>nul
if errorlevel 1 set "FF=0"
if "%FF%"=="1" goto ff_ok
echo   [!]  未找到 FFmpeg / ffprobe。生成 MP3、MP4 前必须安装：
echo       方法一（推荐，免管理员）:  winget install --id Gyan.FFmpeg -e
echo       方法二: 到 https://www.gyan.dev/ffmpeg/builds/ 下载压缩包，
echo               解压后把 bin 目录加入系统 PATH，重开本窗口。
goto ff_done
:ff_ok
for /f "delims=" %%v in ('ffmpeg -version 2^>nul ^| findstr /r /c:"ffmpeg version"') do echo   [OK] FFmpeg  : %%v
:ff_done

rem ---------- 3. 检查 / 安装 Python 依赖 ----------
echo.
"%PY%" -c "import numpy, PIL, faster_whisper" >nul 2>nul
if not errorlevel 1 goto deps_ok

if exist "vendor\faster_whisper" goto try_vendor
goto deps_install

:try_vendor
"%PY%" -c "import numpy, PIL, faster_whisper" >nul 2>nul
if not errorlevel 1 goto deps_ok
goto deps_install

:deps_install
if "%CHECK_ONLY%"=="1" goto deps_missing
echo   正在安装 Python 依赖到 vendor 目录（首次约 300-500MB，请耐心等待）...
"%PY%" -m pip --version >nul 2>nul
if errorlevel 1 goto no_pip
"%PY%" -m pip install --disable-pip-version-check --upgrade --target vendor -r requirements.txt
if errorlevel 1 goto pip_fail
echo   [OK] 依赖安装完成。
"%PY%" -c "import numpy, PIL, faster_whisper" >nul 2>nul
if errorlevel 1 goto pip_fail

rem 可选 CUDA 运行库（有 N 卡才问）
where nvidia-smi >nul 2>nul
if errorlevel 1 goto deps_ok
if exist "vendor\nvidia" goto deps_ok
if exist ".deps\nvidia" goto deps_ok
echo.
echo   检测到 NVIDIA 显卡。额外安装 CUDA 运行库可让转写快 5-10 倍，
echo   但要多下载约 700MB。不装也能用 CPU 慢慢跑。
set "CUDA_ANS="
set /p "CUDA_ANS=现在安装 CUDA 库吗？[y/N]: "
if /i not "%CUDA_ANS%"=="y" goto deps_ok
"%PY%" -m pip install --disable-pip-version-check --upgrade --target vendor nvidia-cublas-cu12 nvidia-cudnn-cu12
if errorlevel 1 echo   [!] CUDA 库安装失败，不影响使用（会自动退回 CPU）。
goto deps_ok

:deps_ok
echo   [OK] 依赖    : numpy / Pillow / faster-whisper 均可正常导入。
"%PY%" -c "import numpy, PIL; import faster_whisper as fw; print('       numpy', numpy.__version__, '| Pillow', PIL.__version__, '| faster-whisper', getattr(fw, '__version__', 'ok'))" 2>nul
goto finish

:deps_missing
echo   [!]  依赖未安装（--check-only 模式不安装）。去掉 --check-only 再运行即可。
goto finish_fail

rem ---------- 收尾 ----------
:finish
echo.
if not "%CHECK_ONLY%"=="1" (
  echo ==================================================
  echo   安装完成。双击 run.bat 即可开始处理语音。
  echo   提示: 首次转写会自动下载 Whisper 模型
  echo         small 约 480MB / medium 约 1.5GB，需要联网。
  echo ==================================================
)
if not "%CHECK_ONLY%"=="1" pause
exit /b 0

:finish_fail
echo.
echo 环境还没装好，请按上面的提示处理后重新运行 setup.bat。
if not "%CHECK_ONLY%"=="1" pause
exit /b 1

:no_python
echo.
echo   [错误] 没有找到带 pip 的 Python 3.10 以上版本。
echo   请先安装 Python 3.10 或更新版本（安装时勾选 Add python.exe to PATH）:
echo       https://www.python.org/downloads/windows/
echo   装完重开这个窗口，再运行 setup.bat。
if not "%CHECK_ONLY%"=="1" pause
exit /b 1

:no_pip
echo.
echo   [错误] 当前 Python 没有 pip，无法安装依赖。
echo   请重新安装 Python 并勾选「Add python.exe to PATH」和 pip 组件，
echo   或手动执行:  "%PY%" -m ensurepip --upgrade
if not "%CHECK_ONLY%"=="1" pause
exit /b 1

:pip_fail
echo.
echo   [错误] 依赖安装失败。常见原因: 网络不通 / 需要代理 / 磁盘空间不足。
echo   可以改用国内镜像重试:
echo       "%PY%" -m pip install --target vendor -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if not "%CHECK_ONLY%"=="1" pause
exit /b 1
