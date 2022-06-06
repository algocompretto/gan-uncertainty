echo "Augmenting data"
python3 src/create_dataset.py

echo "Removing some temporary files"
rm -rf data/temp/np
# rm -rf data/temp/augmented
rm -rf data/temp/generated_binary

echo "Training WGAN algorithm"
python3 src/wgan.py

echo "Sampling TI images generated"
python3 src/ti_sampler.py

echo "Simulating with TI selected"
python3 src/mps.py