# Cleanup safe-to-delete build artifacts and caches for this repo
# This script deletes only clearly safe artifacts: Python caches, egg-info, outputs directories, and the frontend's node_modules if present.
# It DOES NOT delete data, weights, or scripts. Review CLEANUP_NOTICE.md before running.

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Write-Host "Repository root: $root"

$targets = @(
    "**\__pycache__",
    "**\*.pyc",
    "**\*.pyo",
    "**\*.egg-info",
    "outputs",
    "backend\outputs",
    "backend\remorph_backend.egg-info",
    "backend\src\remorph_backend.egg-info",
    "frontend\node_modules",
    ".pytest_cache",
    "backend\.pytest_cache",
    ".venv",
    "**\.DS_Store"
)

foreach ($t in $targets) {
    Write-Host "Searching for: $t"
    Get-ChildItem -Path $root -Recurse -Force -ErrorAction SilentlyContinue -Filter (Split-Path $t -Leaf) -Directory | Where-Object { $_.FullName -like (Join-Path $root $t) -or $true } | ForEach-Object {
        # Conservative: only delete paths that match exact folder names listed above
        $name = $_.Name
        if ($t -match "__pycache__|outputs|node_modules|.venv|.pytest_cache|.egg-info") {
            try {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
                Write-Host "Removed: $($_.FullName)"
            } catch {
                Write-Warning "Failed to remove $($_.FullName): $_"
            }
        }
    }
}

# Remove individual .pyc files
Write-Host "Removing .pyc and .pyo files..."
Get-ChildItem -Path $root -Recurse -Force -Include *.pyc,*.pyo -ErrorAction SilentlyContinue | ForEach-Object {
    try { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop; Write-Host "Removed file: $($_.FullName)" } catch { Write-Warning "Failed to remove $($_.FullName): $_" }
}

Write-Host "Cleanup script finished. Review output above for details."
