param()
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft Huihui Desktop")
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$root = $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Join-Path $root "audio") | Out-Null
Get-Content (Join-Path $root "tts_phrases.txt") -Encoding UTF8 | ForEach-Object {
  if ($_.Trim() -eq "") { return }
  $parts = $_ -split "`t"
  $name = $parts[0]; $text = $parts[1]
  $out = Join-Path $root ("audio/" + $name + ".wav")
  $synth.SetOutputToWaveFile($out, $fmt)
  $synth.Speak($text)
  $synth.SetOutputToNull()
  Write-Host ("OK " + $name + ".wav")
}
$synth.Dispose()
