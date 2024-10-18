# Trajectory DINO
DINO V2 implementation on trajectory data


In here
/data/home/bowen/projects/dinov2_trajectory/dinov2/dinov2/data/datasets/trajectory.py
change the length of the dataset. 


## Visualize Features
Run the following script to visualize features. NOTE: attention visualiztion is not working at the moment.
```
visualize_features.py \
    "--arch", "vit_small", \
    "--pretrained_weights", "...pth", \
    "--data_path", "...npy", \
    "--output_dir", "...", \
    "--feat_type", "pos" 
```
Position embeddings will be reshaped to a square in order to fit neatly. 