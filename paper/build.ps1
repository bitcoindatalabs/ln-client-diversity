# Build main.pdf locally: pdflatex -> bibtex -> pdflatex x2, then summarize warnings.
#   .\build.ps1            build
#   .\build.ps1 -Open      build and open the PDF
#   .\build.ps1 -Clean     delete aux files first (use after changing references.bib)
param([switch]$Open, [switch]$Clean)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# MiKTeX's per-user install is often not on PATH; fall back to its default locations.
$bin = (Get-Command pdflatex -ErrorAction SilentlyContinue).Source | Split-Path -ErrorAction SilentlyContinue
if (-not $bin) {
    $bin = @("$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64", "C:\Program Files\MiKTeX\miktex\bin\x64") |
        Where-Object { Test-Path "$_\pdflatex.exe" } | Select-Object -First 1
}
if (-not $bin) { throw "pdflatex not found. Install MiKTeX or add its bin folder to PATH." }

if ($Clean) {
    Remove-Item main.aux, main.bbl, main.blg, main.log, main.out -ErrorAction SilentlyContinue
}

function Invoke-Step($exe, $arg) {
    # MiKTeX prints notices on stderr ("you have not checked for updates"); judge success by exit code only
    $ErrorActionPreference = "Continue"
    & "$bin\$exe.exe" @arg 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$exe failed (exit $LASTEXITCODE). Last lines of the log:" -ForegroundColor Red
        Get-Content ($(if ($exe -eq "bibtex") { "main.blg" } else { "main.log" })) -Tail 25
        exit 1
    }
}

$tex = @("-interaction=nonstopmode", "-halt-on-error", "-synctex=1", "main.tex")
Invoke-Step pdflatex $tex
Invoke-Step bibtex @("main")
Invoke-Step pdflatex $tex
Invoke-Step pdflatex $tex
# floats can move labels; rerun until references settle (at most twice more)
foreach ($i in 1..2) {
    if (-not (Select-String -Path main.log -Pattern "Rerun to get cross-references" -Quiet)) { break }
    Invoke-Step pdflatex $tex
}

$log = Get-Content main.log
$issues = $log | Select-String -Pattern "undefined|Overfull \\hbox \((\d{2,})|Citation .* undefined|LaTeX Warning: (Reference|There were)"
Write-Host "Built $(Join-Path $PSScriptRoot 'main.pdf')" -ForegroundColor Green
if ($issues) { Write-Host "Warnings worth a look:" -ForegroundColor Yellow; $issues | ForEach-Object { "  $($_.Line)" } }
if ($Open) { Invoke-Item main.pdf }
