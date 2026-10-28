@echo off
chcp 936 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\vendor;%CD%\.deps"

call :find_python
if not defined PY goto no_python
if not "%~1"=="" goto passthrough

rem ---------- 检查依赖提示 ----------
"%PY%" -c "import faster_whisper" >nul 2>nul
if errorlevel 1 (
  echo   [!] 还没装好转写依赖，建议先运行 setup.bat。
  echo       只做「MP3+封面」可以继续，转写/过滤会失败。
  echo.
)

:menu
cls
echo ==================================================
echo    柚子社 Galgame 语音合集  --  一键处理 run.bat
echo ==================================================
echo.
echo    1. 一键处理（转写 + 过滤 + MP3 + MP4）
echo    2. 只转写并生成过滤清单（先看清单再决定）
echo    3. 只生成 MP3 和封面（不生成视频）
echo    4. 生成视频 MP4（附 MP3，用现有转写缓存）
echo    5. 检查已生成的产物（上传前体检）
echo    0. 退出
echo.
set "CHOICE="
set /p "CHOICE=请输入编号后回车: "
if not defined CHOICE goto eof_exit
if "%CHOICE%"=="1" goto do_all
if "%CHOICE%"=="2" goto do_asr
if "%CHOICE%"=="3" goto do_mp3
if "%CHOICE%"=="4" goto do_mp4
if "%CHOICE%"=="5" goto do_qa
if "%CHOICE%"=="0" exit /b 0
echo.
echo   输入无效，请输入 0-5 的数字。
pause
goto menu

rem ---------- 1. 一键 ----------
:do_all
call :ask_folder
call :ask_title
call :ask_strict
call :ask_decisions
echo.
echo 开始处理: %FOLDER%
echo 过滤模式: %STRICT_TEXT%
echo 首次转写可能要 20 分钟到 2 小时，可以放着不管。
echo.
"%PY%" -m voicepack all "%FOLDER%" %TITLEARG% %STRICTARG% %DECIARG%
set "RC=%errorlevel%"
call :reset_console
if not "%RC%"=="0" goto run_failed
goto run_ok

rem ---------- 2. 只转写 + 过滤清单 ----------
:do_asr
call :ask_folder
echo.
"%PY%" -m voicepack asr "%FOLDER%"
set "RC=%errorlevel%"
call :reset_console
if not "%RC%"=="0" goto run_failed
"%PY%" -m voicepack filter "%FOLDER%"
set "RC=%errorlevel%"
call :reset_console
if not "%RC%"=="0" goto run_failed
echo.
echo 清单已生成: %FOLDER%\voicepack_out\filter_report.csv
echo 用 Excel 打开，把 action 列改成 drop 或 keep 后保存；
echo 再选 1 / 3 / 4，脚本会问你要不要用这份人工清单（填它的路径即可）。
goto run_ok

rem ---------- 3. 只出 MP3 ----------
:do_mp3
call :ask_folder
call :ask_title
call :ask_strict
call :ask_decisions
echo.
"%PY%" -m voicepack build "%FOLDER%" --mp3-only %TITLEARG% %STRICTARG% %DECIARG%
set "RC=%errorlevel%"
call :reset_console
if not "%RC%"=="0" goto run_failed
goto run_ok

rem ---------- 4. 出 MP4 ----------
:do_mp4
call :ask_folder
call :ask_title
call :ask_strict
call :ask_decisions
echo.
"%PY%" -m voicepack build "%FOLDER%" %TITLEARG% %STRICTARG% %DECIARG%
set "RC=%errorlevel%"
call :reset_console
if not "%RC%"=="0" goto run_failed
goto run_ok

rem ---------- 5. 质检 ----------
:do_qa
call :ask_folder
set "OUTDIR=%FOLDER%\voicepack_out"
set "OUTI="
set /p "OUTI=产物目录（直接回车 = %OUTDIR% ）: "
if defined OUTI set "OUTDIR=%OUTI%"
set "MINS="
set /p "MINS=整批总时长（分钟，可留空跳过核对）: "
set "MINARG="
if defined MINS set "MINARG=--expected-minutes %MINS%"
echo.
"%PY%" tools\qa_check.py "%OUTDIR%" %MINARG%
set "RC=%errorlevel%"
call :reset_console
echo.
pause
goto menu

rem ---------- 结果 ----------
:run_ok
echo.
echo [完成] 产物默认在: %FOLDER%\voicepack_out
echo        建议打开 filter_report.csv 人工过一遍待确认项。
set "OPENIT="
set /p "OPENIT=现在打开输出目录吗？[Y/n]: "
if not defined OPENIT goto run_end
if /i "%OPENIT%"=="n" goto run_end
start "" explorer "%FOLDER%\voicepack_out"
:run_end
echo.
pause
goto menu

:run_failed
echo.
echo [失败] 上一步出错了，请把上面的错误信息截图反馈。
echo        常见原因: 没有安装 FFmpeg / 网络断了 / 磁盘不够。
pause
goto menu

:pause_ret
pause
goto menu

rem ---------- 子过程 ----------
:ask_folder
echo.
set "FOLDER="
set /p "FOLDER=语音文件夹路径（可把文件夹拖进来，或粘贴 E:\mrxyy）: "
if not defined FOLDER goto eof_exit
set "FOLDER=%FOLDER:"=%"
if "%FOLDER:~-1%"=="\" set "FOLDER=%FOLDER:~0,-1%"
if "%FOLDER%"=="" goto ask_folder
if not exist "%FOLDER%\" goto ask_folder_bad
exit /b 0
:ask_folder_bad
echo   [!] 找不到文件夹: %FOLDER%
goto ask_folder

:ask_title
set "TITLE="
set "TITLEARG="
set /p "TITLE=角色名 / 标题（直接回车用文件夹名）: "
if defined TITLE set "TITLEARG=--title "%TITLE%""
exit /b 0

rem ---------- 人工清单（可选） ----------
:ask_decisions
set "DECI="
set "DECIARG="
set "DREPORT=%FOLDER%\voicepack_out\filter_report.csv"
if not exist "%DREPORT%" goto ad_done
echo.
echo   发现上次的过滤清单: %DREPORT%
set /p "DECI=人工修改过的清单路径（直接回车 = 不使用）: "
if not defined DECI goto ad_done
if not exist "%DECI%" (
  echo   [!] 文件不存在，忽略: %DECI%
  set "DECI="
  goto ad_done
)
set "DECIARG=--decisions "%DECI%""
:ad_done
exit /b 0

:ask_strict
set "STRICTARG="
set "STRICT_TEXT=默认（自动剔除高可疑句，保留待确认句）"
set "STRICT="
set /p "STRICT=严格模式？上传前推荐开启，待确认句也剔除 [y/N]: "
if /i "%STRICT%"=="y" set "STRICTARG=--strict"
if /i "%STRICT%"=="y" set "STRICT_TEXT=严格（待确认句也剔除）"
exit /b 0

:find_python
set "PY="
if not defined VOICEPACK_PYTHON goto fp_venv
if not exist "%VOICEPACK_PYTHON%" goto fp_venv
"%VOICEPACK_PYTHON%" -c "import numpy, PIL" >nul 2>nul
if not errorlevel 1 set "PY=%VOICEPACK_PYTHON%"
if defined PY exit /b 0
:fp_venv
if not exist "%CD%\.venv\Scripts\python.exe" goto fp_dsh
"%CD%\.venv\Scripts\python.exe" -c "import numpy, PIL" >nul 2>nul
if not errorlevel 1 set "PY=%CD%\.venv\Scripts\python.exe"
if defined PY exit /b 0
:fp_dsh
if not exist "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" goto fp_path
"%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" -c "import numpy, PIL" >nul 2>nul
if not errorlevel 1 set "PY=%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if defined PY exit /b 0
:fp_path
python -c "import numpy, PIL" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY exit /b 0
py -3 -c "import numpy, PIL" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
exit /b 0

:passthrough
"%PY%" -m voicepack %*
exit /b %errorlevel%

:no_python
echo.
echo   [错误] 没找到可用的 Python 环境（需要 numpy 和 Pillow）。
echo   请先双击运行 setup.bat 安装环境。
pause
exit /b 1

:reset_console
rem voicepack CLI switches console codepage to 65001; switch back to GBK for the next prompts
chcp 936 >nul
exit /b 0

:eof_exit
rem 输入被关闭（控制台结束/管道结束）时直接结束整个脚本；
rem 这里不能用 exit /b，因为在 call 的子过程里 exit /b 只会返回上一层。
exit
