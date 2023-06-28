for f in "/data/nvme3/checkpoint/BELLE-LLaMA-EXT-13B"/*; \
    do if [ -f "$f" ]; then \
       python scripts/decrypt.py "$f" "/data/nvme3/checkpoint/consolidated.00.pth" "/home/tyang/BELLE-LLaMA-EXT-13B"; \
    fi; \
done