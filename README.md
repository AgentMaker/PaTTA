# Patta
Image Test Time Augmentation with Paddle2.0!

```
           Input
             |           # input batch of images 
        / / /|\ \ \      # apply augmentations (flips, rotation, scale, etc.)
       | | | | | | |     # pass augmented batches through model
       | | | | | | |     # reverse transformations for each batch of masks/labels
        \ \ \ / / /      # merge predictions (mean, max, gmean, etc.)
             |           # output batch of masks/labels
           Output
```
## Table of Contents
1. [Quick Start](#quick-start)
2. [Transforms](#transforms)
3. [Aliases](#aliases)
4. [Merge modes](#merge-modes)
5. [Installation](#installation)

## Quick start (Default Transforms)

#####  Segmentation model wrapping [[docstring](patta/wrappers.py#L8)]:
```python
import patta as tta
tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
```
#####  Classification model wrapping [[docstring](patta/wrappers.py#L52)]:
```python
tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform())
```

#####  Keypoints model wrapping [[docstring](patta/wrappers.py#L96)]:
```python
tta_model = tta.KeypointsTTAWrapper(model, tta.aliases.flip_transform(), scaled=True)
```
**Note**: the model must return keypoints in the format `Tensor([x1, y1, ..., xn, yn])`

## Advanced Examples (DIY Transforms)
#####  Custom transform:
```python
# defined 2 * 2 * 3 * 3 = 36 augmentations !
transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
)

tta_model = tta.SegmentationTTAWrapper(model, transforms)
```
##### Custom model (multi-input / multi-output)
```python
# Example how to process ONE batch on images with TTA
# Here `image`/`mask` are 4D tensors (B, C, H, W), `label` is 2D tensor (B, N)

for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform() 
    
    # augment image
    augmented_image = transformer.augment_image(image)
    
    # pass to model
    model_output = model(augmented_image, another_input_data)
    
    # reverse augmentation for mask and label
    deaug_mask = transformer.deaugment_mask(model_output['mask'])
    deaug_label = transformer.deaugment_label(model_output['label'])
    
    # save results
    labels.append(deaug_mask)
    masks.append(deaug_label)
    
# reduce results as you want, e.g mean/max/min
label = mean(labels)
mask = mean(masks)
```
 
## Optional Transforms
  
| Transform      | Parameters                | Values                            |
|----------------|:-------------------------:|:---------------------------------:|
| HorizontalFlip | -                         | -                                 |
| VerticalFlip   | -                         | -                                 |
| Rotate90       | angles                    | List\[0, 90, 180, 270]            |
| Scale          | scales<br>interpolation   | List\[float]<br>"nearest"/"linear"|
| Resize         | sizes<br>original_size<br>interpolation   | List\[Tuple\[int, int]]<br>Tuple\[int,int]<br>"nearest"/"linear"|
| Add            | values                    | List\[float]                      |
| Multiply       | factors                   | List\[float]                      |
| FiveCrops      | crop_height<br>crop_width | int<br>int                        |
 
## Aliases (Combos)

  - flip_transform (horizontal + vertical flips)
  - hflip_transform (horizontal flip)
  - d4_transform (flips + rotation 0, 90, 180, 270)
  - multiscale_transform (scale transform, take scales as input parameter)
  - five_crop_transform (corner crops + center crop)
  - ten_crop_transform (five crops + five crops on horizontal flip)
  
## Merge modes
 - mean
 - gmean (geometric mean)
 - sum
 - max
 - min
 - tsharpen ([temperature sharpen](https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107716#latest-624046) with t=0.5)
 
## Installation
PyPI:
```bash
# After downloading the whole dir
$ git clone https://github.com/AgentMaker/PaTTA.git
$ pip install PaTTA-mian/

# or

$ pip install git+https://github.com/AgentMaker/PaTTA.git
```

## Run tests

```bash
# run test_transforms.py and test_base.py for test
python test/test_transforms.py
python test/test_base.py
```