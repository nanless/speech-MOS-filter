# speech-MOS-filter
Filtering speech sentences using multiple MOS scoring tools

## Usage

```
python compute_mos.py --wav_files_dir /path/to/wavfile_directory --output_statistics_dir /path/to/output_statistics_directory
python hisgram.py --statistics_folder_path /path/to/output_statistics_directory
python filter.py --statistics_folder_path /path/to/output_statistics_directory --origianl_folder_path /path/to/original_wavfile_directory --filtered_folder_path /path/to/filtered_wavfile_directory
```
You can set the threshold values for each MOS scoring tool in the `filter.py` file.

## Credits

Special thanks to the following projects:

[DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS)

[NISQA](https://github.com/gabrielmittag/NISQA)

[SIGMOS](https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos)

[UTMOS](https://github.com/sarulab-speech/UTMOS22)

[WVMOS](https://github.com/AndreevP/wvmos)

