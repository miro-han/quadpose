Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
script = fso.BuildPath(folder, "Start_Camera_UI.cmd")
shell.CurrentDirectory = folder
shell.Run Chr(34) & script & Chr(34), 1, False
