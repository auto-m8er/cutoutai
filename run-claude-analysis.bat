@echo off
echo ============================================
echo  CutoutAI Analysis via Claude Code + CCR
echo ============================================
echo.

:: Set environment for CCR
set CLAUDE_BASE_URL=http://127.0.0.1:3456
set ANTHROPIC_BASE_URL=http://127.0.0.1:3456

:: Change to project directory
cd /d "C:\Users\jonat_cau4\.gemini\antigravity\scratch\background removal tool"

echo Current directory: %CD%
echo CCR URL: %CLAUDE_BASE_URL%
echo.
echo Starting Claude Code with analysis prompt...
echo.

:: Run Claude Code with the analysis task
claude -p "Read and analyze all files in this project: PROMPT.md, cutoutai.py, api.py, @fix_plan.md, specs/requirements.md, and README.md. Provide a COMPREHENSIVE ANALYSIS including: 1) Code quality assessment, 2) Edge handling for t-shirt mockups - is thresholding correct?, 3) Multi-element capture (bubbles) - is the threshold low enough?, 4) API robustness for n8n/Make integration, 5) Startup preloading implementation, 6) Any bugs or issues found. Then provide SPECIFIC IMPROVEMENT RECOMMENDATIONS with code examples. After analysis, ask if I want you to implement the improvements."

pause
