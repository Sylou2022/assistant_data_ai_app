@echo off
REM ============================================================
REM  Assistant Data AI - lancement en un double-clic
REM  Utilise le Python du projet (.venv) et ouvre le navigateur
REM  automatiquement sur http://localhost:7860 quand tout est prêt.
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    echo [!] Environnement .venv introuvable : utilisation du Python du systeme.
    set "PY=python"
)

set "OPEN_BROWSER=1"
set "PYTHONIOENCODING=utf-8"
echo Demarrage d'Assistant Data AI... (la premiere connexion a la base peut prendre jusqu'a 1 minute)
"%PY%" -m app.server

echo.
echo Le serveur s'est arrete. Appuyez sur une touche pour fermer cette fenetre.
pause >nul
