param([string]$Script = "eval_megadepth.py", [string]$Extra = "")
$py = ".\.venv\Scripts\python.exe"
   $groups = @("KAZE", "ALIKED XFeat", "SIFT GFTT_SIFT", "AKAZE STAR_BRIEF", "ORB ORB_BEBLID ORB_FREAK BRISK", "FAST_BRIEF FAST_FREAK GFTT_BRIEF")
foreach ($g in $groups) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; & '$py' scripts\$Script --presets $g $Extra"
}