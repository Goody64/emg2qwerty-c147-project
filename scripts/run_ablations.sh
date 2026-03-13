#!/bin/bash
# Run augmentation and data ablation experiments for the writeup.
# Each training run takes ~40 min. Run sequentially.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== 1. Augmentation ablation: TDS without SpecAugment ==="
python -m emg2qwerty.train model=tds_conv_ctc transforms=no_specaug trainer.max_epochs=80
# Checkpoint saved to logs/.../checkpoints/; note test CER from output

echo ""
echo "=== 2. Data ablation: 4 sessions ==="
python -m emg2qwerty.train model=tds_conv_ctc user=single_user_4sessions trainer.max_epochs=80

echo ""
echo "=== 3. Data ablation: 8 sessions ==="
python -m emg2qwerty.train model=tds_conv_ctc user=single_user_8sessions trainer.max_epochs=80

echo ""
echo "=== 4. Data ablation: 12 sessions ==="
python -m emg2qwerty.train model=tds_conv_ctc user=single_user_12sessions trainer.max_epochs=80

echo ""
echo "Done. Update writeup Tables 5 and 6 with the test CER from each run."
