echo "[INFO] Launching dataset creator script"
python src/create_dataset.py
echo "[INFO] Successfully finished scripting"

# Loop to know which model to use
select model in aae dcgan wgan
do
    echo "Model selected: $model"
    break
done

# Running deep generative model
python src/generative_models/$model/$model.py

# Deleting temporary files
echo "[INFO] Cleaning up temporary files"
rm -rf data/temp/