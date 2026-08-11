# Hello!
There is my Computer Vision project. In this project i made model which detects people`s heads, based on YOLO architecture. To train this model i created my own dataset that contains +- 667 images.

 In practice this model can be used in military detection systems to detect targets or in security systems to detect criminals.
## To start using this model you have to do several steps.
1. download dependencies by `pip install -r requirements.txt`
2. import model from model_usage.py by 
```python
from model_usage import model
```
   
3. download model weights by add this code in your project
``` python
state_dict = torch.load('weights_path')
model.load_state_dict(state_dict)
```
4. to handle image you need to use function get_bboxes that situated in model_usage by
```python
from model_usage import get_bboxes
```
All work is done !
So to get bounding boxes you need to specify image (h,w,3) and thresholds in `get_bboxes()`.
