# Image Enhancer Prototype

## Installation


Before running the script, install all the requirements in the `requirements.txt`.

```bash
pip install -r requirements.txt
```

###### _To note that the script was only tested in Python3.7_

## Usage

```bash
python image_enhancer.py <image-path> <methods-list> <destination-path>
```

#### image-path
+ Complete or relative path to the image.

#### methods-list 
Selected methods from the list bellow, in the desired order, separated by a single comma. 
+ CS: Contrast Stretching
+ HE: Histogram Equalizer
+ CL: Contrast Limited Adaptive Histogram equalizer
+ GC: Gamma Correction
+ NL: Non-local Means Denoising
+ UM: Unsharp Masking

#### image-path 
+ Complete path where the new image will be saved. If there is no path, the image will be only presented and not save.

#### Use Example
```bash
python image_enhancer.py C:/User/Documents/Lenna.jpg CL,UM,NL C:/User/Documents
```
#### Test on the Data-set
To test the data-set run
```bash
python image_enhancer.py -TEST <methods-list> <sub-set>
```
Where _sub-set_ can be:
  + bad
  + medium
  + good
  + all

Make sure that the data-set is the correct one, and is in the same directory as the script.

## Data-Set
Link to data-set with test results:
https://drive.google.com/open?id=1S-wCfn6u-1SNgZ9zhWhSmdHIinZvLVBt
