echo "Augmenting data"
python3 src/create_dataset.py

echo "Training WGAN algorithm"
python3 src/wgan.py

echo "Sampling TI images generated"
python3 src/ti_sampler.py

echo "Simulating with TI selected"
python3 src/mps.py
