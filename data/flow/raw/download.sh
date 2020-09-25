remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing Cytometry Flow Dataset..."
  rm *.fcs
  echo "Removed!"
else
  echo "Downloading Cytometry Flow Dataset..."
  wget "https://flowrepository.org/experiments/102/fcs_files/10673/download" -O pbmc_luca.fcs
  echo "Download Finished!"
fi