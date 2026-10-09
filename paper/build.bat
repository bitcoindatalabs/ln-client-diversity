@echo off
rem Build main.pdf from any terminal (cmd or PowerShell), bypassing the "running scripts is disabled" policy.
rem   build          build
rem   build -Open    build and open the PDF
rem   build -Clean   delete aux files first (use after changing references.bib)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
