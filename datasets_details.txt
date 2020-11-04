The dataset paths tree is organized as follow:

--> oxford60k/  
 |--> image/  
 | |--> train/  
 | | |--> gallery/ (train gallery from SVOX domain)  
 | | |--> queries/ (train queries from SVOX domain)  
 | | |--> queries_**x**/ where **x** in [1, 5] (train queries from target domain)  
 | | |--> queries__queries_biost_few_**x**/ where **x** in [1, 5] (train queries from SVOX + pseudo-target domains)  
 | | |--> queries_n5_d**x**/ where **x** in [1, 5] (just 5 images randomly sampled from queries of the target domain)  
 | |--> val/  
 | | |--> gallery/ (val gallery from SVOX domain)  
 | | |--> queries/ (val queries from SVOX domain)  
 | | |--> queries_biost_few_**x**/ where **x** in [1, 5] (val queries from pseudo-target domain)  
 | |--> test  
 | | |--> gallery/ (test gallery from SVOX domain)  
 | | |--> queries/ (test queries from SVOX domain)  
 | | |--> queries_**x**/ where **x** in [1, 5] (test queries from target domain)  
  
**x** is the SCENARIO number = [1 - Sun, 2 - Rain, 3 - Snow, 4 - Night, 5 - Overcast] .  
