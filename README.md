# SODT_ResNet  
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152 are respectively used as bacbone to construct object detection model based on Faster RCNN  

Code for paper "[Deer survey from drone thermal imagery using enhanced faster R-CNN based on ResNets and FPN](https://doi.org/10.1016/j.ecoinf.2023.102383)"  

> This paper is supported by a NSF project "Feedbacks Between Human Community Dynamics and Sociobiological Vulnerability in a Biodiversity Hotspot" (BCS-1211498)  
Due to the restrictions of the confidentiality agreement, we are unable to fully disclose all thermal image datasets. However, within the limits allowed by the agreement, we are sharing a portion of the thermal images to construct a training dataset for validating the models proposed in this paper. Consequently, the performance of models trained with these datasets may be lower compared to the models discussed in this paper.  

> All codes are in demo.ipynb  

## Citation
If you use this code in your work, please cite our [paper](https://doi.org/10.1016/j.ecoinf.2023.102383):  
```
@article{
    title = {Deer survey from drone thermal imagery using enhanced faster R-CNN based on ResNets and FPN},
    journal = {Ecological Informatics},
    volume = {79},
    year = {2024},
    issn = {1574-9541},
    doi = {https://doi.org/10.1016/j.ecoinf.2023.102383},
    url = {https://www.sciencedirect.com/science/article/pii/S1574954123004120},
    author = {Haitao Lyu, Fang Qiu, Li An, Douglas Stow, Rebecca Lewison, Eve Bohnett}
    }
```

## Model Architecture  
![](https://ars.els-cdn.com/content/image/1-s2.0-S1574954123004120-gr3_lrg.jpg)  

## Detection Examples  
![](https://ars.els-cdn.com/content/image/1-s2.0-S1574954123004120-gr14_lrg.jpg)  

![](https://ars.els-cdn.com/content/image/1-s2.0-S1574954123004120-gr15_lrg.jpg) 
> Fig. 14,15. (a) denotes the original image with the ground-true bounding boxes marked in red. (b) denotes the output of FRC_ResNet18FPN. (C) denotes the output of FRC_ResNet34FPN. (d) denotes the output of FRC_ResNet50FPN. (e) denotes the output of FRC_ResNet101FPN. (f) denotes the output of FRC_ResNet152FPN.

![](https://ars.els-cdn.com/content/image/1-s2.0-S1574954123004120-gr17_lrg.jpg)
> Fig. 17. (a) The detection results from the model using big-scale anchor boxes. (b) The detection results from the model using small-scale anchor boxes. The three red arrows point at the objects missed by the model in (a).  

## License
SODT_ResNet follows [`CC BY-NC-SA-4.0`](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/LICENSE.md) license and is thus open source and freely available for academic research purposes only, no commercial use is permitted.  


