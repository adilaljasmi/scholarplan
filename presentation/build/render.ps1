# Render every slide of the deck to PNG using the installed PowerPoint (visual QA).
param([string]$Deck, [string]$OutDir)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutDir | Out-Null
Get-ChildItem $OutDir -Filter *.png -ErrorAction SilentlyContinue | Remove-Item -Force -Confirm:$false
$ppt = New-Object -ComObject PowerPoint.Application
try {
  $pres = $ppt.Presentations.Open((Resolve-Path $Deck).Path, $true, $false, $false)
  $i = 1
  foreach ($slide in $pres.Slides) {
    $slide.Export((Join-Path (Resolve-Path $OutDir).Path ("slide-{0:D2}.png" -f $i)), "PNG", 1600, 900)
    $i++
  }
  $pres.Close()
} finally {
  $ppt.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
Get-ChildItem $OutDir -Filter *.png | Select-Object Name, Length
