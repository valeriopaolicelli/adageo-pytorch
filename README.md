# AdAGeo: Adaptive-Attentive Geolocalization from few queries: a hybrid approach  
  
**[13th-Jun-2022] News:** Published AdaGeo-Lite ([article](https://www.frontiersin.org/articles/10.3389/fcomp.2022.841817/full), [code](https://github.com/valeriopaolicelli/adageo-WACV2021/tree/fda))  

PyTorch code of **AdAGeo: Adaptive-Attentive Geolocalization from few queries: a hybrid approach**.

Short presentation (WACV2021): https://www.youtube.com/watch?v=URQCLkDIygM

![Architecture](architecture.jpg)


Requirements:
*  Python 3.4+
*  Pip3
*  `pip3 install -r *requirements.txt*`
  
  
Datasets and ResNet18-based AdAGeo are available on request.
Datasets details are provided in *datasets_details.txt*.

Please, set the parameter `--allDatasetsPath` in **const.py** before starting. It is the root path (hardcoded) where *oxford60k* is located.  

Train model for a certain target SCENARIO:  
*  Phase 1: Check [AdaGeo-Lite](https://github.com/valeriopaolicelli/adageo-WACV2021/tree/fda) for the quick creation of a pseudo-target dataset.  
*  Phase 2: Training starting from ResNet18 pretrained on Places365 (code already provides model/weights downloading from project https://github.com/CSAILVision/places365) with default parameters set to the ones declared in our paper.  
Using our dataset paths management, you only need to decide the SCENARIO (int value) of Oxford RobotCar dataset and run the command below.  
`python main.py --expName={what you want} --attention --trainQ=train/queries__queries_biost_few_{SCENARIO} --valQ=val/queries_biost_few_{SCENARIO} --testQ=test/queries_{SCENARIO} --grl --grlDatasets=train/queries+train/queries_biost_few_{SCENARIO}+train/queries_n5_d{SCENARIO} --epochDivider=4 --patience=3 `  
  
Test model for a certain target SCENARIO:  
`python eval.py --expName=<what you want> --resume=<path to trained model> --ckpt=best --attention --testQ=test/queries_{SCENARIO}`  
  
BibTex:
@ARTICLE{PAOLICELLI-2022-FRONTIERS,
  AUTHOR={Paolicelli, Valerio and Berton, Gabriele and Montagna, Francesco and Masone, Carlo and Caputo, Barbara},   
  TITLE={Adaptive-Attentive Geolocalization From Few Queries: A Hybrid Approach},      
  JOURNAL={Frontiers in Computer Science},      
  VOLUME={4},           
  YEAR={2022},      
  URL={https://www.frontiersin.org/articles/10.3389/fcomp.2022.841817},       
  DOI={10.3389/fcomp.2022.841817},
  ISSN={2624-9898}
}
