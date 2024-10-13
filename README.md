<!-- pip install git+https://github.com/deepmind/surface-distance.git -->

# Pipeline
The main idea of our method is learning a recommendation agent to recommend the next slice for user to interact with.

The pipleline is to preduce the reward for the agent first, then pretrain agent, ensuring that the agent has the memory pool. 
Then, we train the agent and validate it.

The base models should be trained and validated separately. Once the checkpoints for them are stored, they can be used to train and test the agent.

## Datasets

Our code support training unionly or separately.

Our dataset is organized as follow:
```bash
-data
    -train
        -tumor
            -colon
                -imagesTr
                    -colon_train_001.nii.gz
                    - ...
                    -colon_train_085.nii.gz
                -labelsTr
            -kits
                -imagesTr
                -labelsTr
            -lits
                -imagesTr
                -labelsTr
            -pancreas
                -imagesTr
                -labelsTr
    -validation
        -tumor
            -colon
                -imagesTr
                -labelsTr
            -kits
                -imagesTr
                -labelsTr
            -lits
                -imagesTr
                -labelsTr
            -pancreas
                -imagesTr
                -labelsTr

```

You may also udpate the parameters in corresponding python files. Be aware that the path should be replaced to yours.


## Usage

Our method consists of two components, i.e., a 3D interactive medical segmentaiton model, which performs segmentation based on given 3D click prompts,  and an Agent for recommonding 2D slices for users to interact with.  Our method mainly focus on the second part, i.e., how to design an effective Agent to recommond 2D slices, faclicting efficient user interaction.

### Interactive segmentation models (SAM-Med3d)

For the 3D interative segmentation model, generally we can use any interactive segmentation models, here we choose SAM-Med3D as our segmentor considering its great ISeg capability and efficient framework.

You can choose to train the SAM-Med3d with your own data, following are some envs requirement, training, and inference instructions, or download original pretrained weights from [here](https://github.com/uni-medical/SAM-Med3D#-checkpoint) and skip this component.

```bash
conda create --name medrl python=3.7 
conda activate medrl
pip install light-the-torch && ltt install torch
pip install torchio opencv-python-headless matplotlib prefetch_generator monai edt medim
pip install -r requirements.txt
```

Then, train the SAM-Med3D model
```bash
bash train.sh
```

To validate the model, run:
```bash
bash infer.sh
```

The parameters can be adjusted in the `train.py`

### Agent models

The training of agents consist of two stages: (1) producing reward and pretrain agent (2) train the agent model.


producing reward and pretrain agent using these two commonds:

```bash
python recommendation/produce_reward.py
python recommendation/pretrain_agent.py
```

To train the agent and inference, following:

```bash
python recommendation/train_agent.py
python recommendation/test_agent.py
```

To use different dataset, parameters can be added in the pretrain_agent.py and train_agent.py

