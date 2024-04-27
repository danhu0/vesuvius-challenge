# vesuvius-challenge
Our implementation of ink detection in the Herculaneum papyri

here's the repo/clone link for the winning implementation that we're basing the project on:
https://github.com/Bodillium/Vesuvius-Grandprize-Winner.git. This is a fork off the original
grand prize implementation of the Vesuvius Project. This also uses VesuviusDataDownload
https://github.com/JamesDarby345/VesuviusDataDownload.git, for easier segment downloading.


Running this locally takes a lot a lot of space, so it's best done on a department machine. 
To run on a department machine, you will need to make a venv. This can be done with the command 
```python3 -m venv name-of-venv```. Source your venv and then cd into the Vesuvius-Grandprize-Winner
dir. This will dump most of the required dependencies into the new venv, but you will need to get rclone
manually.

To do this, go to the debian linux rclone link from the website and manually download it. Extract the contents
of the folder (specifically extract the data.tar.gz file) and then extract the tarball of this file using
tar -xzf data.tar.gz, which gives you the binary executable. Drop this into the bin of your venv and check
rclone works with ```rclone --verison```. Now you can run the ```./download.sh``` script.



clean up the results of the download, it borked up 20231210121321
