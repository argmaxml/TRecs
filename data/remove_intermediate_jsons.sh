ls -1 | grep -v npy | grep -v meta | xargs -d "\n" -I {} rm "{}"
