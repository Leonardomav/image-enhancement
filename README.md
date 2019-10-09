# Image Enhancer Prototype

## Installation


Before running the script, install all the requirements in the `requirements.txt`.

```bash
pip install -r requirements.txt
```

###### _To note that the script was only tested in Python3.7_

## Usage

```bash
python main.py <image-path> <methods-list> <destination-path>
```

### <image-path> 
+ Complete or relative path to the image.

### <methods-list> 
Selected methods from the list bellow, in the desired order, separated by a comma. 
+ CS: Contrast Stretching
+ HE: Histogram Equalizer
+ CL: Contrast Limited Adaptive Histogram equalizer
+ GC: Gamma Correction
+ FD: Denoising using _fastn1_
+ UM: Unsharp Masking

### <image-path> 
+ Complete path where the new image will be saved. If there is no path, the image will be only presented and not save.

Use Example
```bash
python main.py "C:\User\Documents\Lenna.jpg" "CL,UM,FD" "C:\User\Documents"
```
