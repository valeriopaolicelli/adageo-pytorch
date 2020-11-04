# AdAGeo: Adaptive-Attentive Geolocalization from few queries: a hybrid approach
Pytorch code of AdAGeo - WACV2021

Arxiv paper: https://arxiv.org/abs/2010.06897 accepted at WACV2021

![Architecture](architecture.jpg)


Requirements:
*  Python 3.4+
*  Pip3
*  `pip3 install -r requirements.txt`
  
  
Datasets and ResNet18-based AdAGeo are available on request.
Datasets are organized as follow:  
--> oxford60k/  
 |  
 |--> image/  
 | |   
 | |--> train/  
 | | |  
 | | |--> gallery/ (train gallery from SVOX domain)  
 | | |--> queries/ (train queries from SVOX domain)  
 | | |--> queries_**x**/ where **x** in [1, 5] (train queries from target domain)  
 | | |--> queries__queries_biost_few_**x**/ where **x** in [1, 5] (train queries from SVOX + pseudo-target domains)  
 | | |--> queries_n5_d**x**/ where **x** in [1, 5] (just 5 images randomly sampled from queries of the target domain)  
 | |  
 | |--> val/  
 | | |  
 | | |--> gallery/ (val gallery from SVOX domain)  
 | | |--> queries/ (val queries from SVOX domain)  
 | | |--> queries_biost_few_**x**/ where **x** in [1, 5] (val queries from pseudo-target domain)  
 | |  
 | |--> test  
 | | |  
 | | |--> gallery/ (test gallery from SVOX domain)  
 | | |--> queries/ (test queries from SVOX domain)  
 | | |--> queries_**x**/ where **x** in [1, 5] (test queries from target domain)  
  
**x** is the SCENARIO number = [1 - Sun, 2 - Rain, 3 - Snow, 4 - Night, 5 - Overcast] .  
Please, set the parameter `--allDatasetsPath` in **const.py** before starting. It is the root path (hardcoded) where *oxford60k* is located.  

Train model for a certain Oxford RobotCar SCENARIO:  
*  Phase 1:  
*  Phase 2: Training starting from ResNet18 pretrained on Places365 (code already provides model/weights downloading from project https://github.com/CSAILVision/places365) with default parameters set to the ones declared in our paper.  
Using our dataset paths management, you only need to decide the SCENARIO (int value) of Oxford RobotCar dataset and run the command below.  
`python main.py --expName={what you want} --attention --trainQ=train/queries__queries_biost_few_{SCENARIO} --valQ=val/queries_biost_few_{SCENARIO} --testQ=test/queries_{SCENARIO} --grl --grlDatasets=train/queries+train/queries_biost_few_{SCENARIO}+train/queries_n5_d{SCENARIO} --epochDivider=4 --patience=3 `  
  
Test model for a certain Oxford RobotCar SCENARIO:  
`python eval.py --expName=<what you want> --resume=<path to trained model> --ckpt=best --attention --testQ=test/queries_{SCENARIO}`  
