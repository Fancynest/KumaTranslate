Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "d:\HQ-Gabriel\Code\manga_translate"
WshShell.Run "powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File ""d:\HQ-Gabriel\Code\manga_translate\launcher.ps1""", 0, False
