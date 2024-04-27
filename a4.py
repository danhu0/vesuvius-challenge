import os

# Define the base directory where the directories will be created
base_dir = 'vesuvius_model'

# Where to put the checkpoints used in inference
checkpoints_dir = os.path.join(base_dir, 'checkpoints')
os.makedirs(checkpoints_dir, exist_ok=True)

# Where the inference outputs (predictions) are saved
inference_output_dir = os.path.join(base_dir, 'inference_output')
os.makedirs(inference_output_dir, exist_ok=True)

# The segment on which inference is to be run
inference_segment_dir = os.path.join(base_dir, 'volume/segments/20231005123336/layers')
os.makedirs(inference_segment_dir, exist_ok=True)

# The segments used to train the model
train_segments = [
    '20230702185753', '20230929220926', '20231005123336', '20231007101619',
    '20231012184423', '20231016151002', '20231022170901', '20231031143852',
    '20231106155351', '20231210121321', '20231221180251', '20230820203112'
]

for segment in train_segments:
    train_segment_dir = os.path.join(base_dir, f'training/train_scrolls/{segment}/layers')
    os.makedirs(train_segment_dir, exist_ok=True)

# Where the training outputs (checkpoints, etc) are saved
training_outputs_dir = os.path.join(base_dir, 'training/outputs')
os.makedirs(training_outputs_dir, exist_ok=True)
