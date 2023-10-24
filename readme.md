# Demo for the JSS23 Paper

![257033126-835d9fe9-17eb-4dc3-b4ad-f62046ce7e33](https://github.com/escapar/JSS23_demo/assets/13232736/c94b53f4-2a78-4905-94b0-dd25ba12bc4f)

## Description

This is a webapp demo repository for XAI application in code smell prioritization.    
This repository contains model generation, dataset generation, and data analyzing functions.    

## Demos Videos
**Two video demos containing the execution result of the program is available in the ./vids/ directory.**    
https://github.com/escapar/JSS23_demo/blob/master/vids/Dataset%20Generation.mp4       
https://github.com/escapar/JSS23_demo/blob/master/vids/Result%20Analyze.mp4       

## Approach  
Since we do not have the original code and toolset of MSR'20 [1], we are replicating the approach using similar techniques and datasets.     
We are using the Blob smell in the MLCQ dataset annotated by third-party developers for dataset generation.      
Afterward, CK Metrics [2] are exploited to generate code metrics. The process metrics are replicated by ourselves using the PyDriller tool [3].      
Then, we follow the process described in our paper to generate predictions and explanations.     

## Usage
1. Install the required dependencies.      
```
pip install -r requirement.txt
```
2. Execute Flask App.      
```
flask --app main run
```
3. Access http://localhost:5000/     
4. Type in the URL of the Git repo for feature generation (see **./vids/Dataset Generation.mp4** for a video example), and wait for it to finish. You may check the console to see the process. This process may take some time because we are using a synchronical solution to avoid consuming too much storage. It could also be done asynchronically, if each time we copy an entire project for each analysis to avoid conflict.     
5. Access http://localhost:5000/project/{name_of_your_repo} to analyze data (see **./vids/Result Analyze.mp4** for a video example). The predictions are made instantly (when you click the select dropdown).       

## Author
Zijie Huang, East China University of Science and Technology
https://hzjdev.github.io/

## References 

[1] Fabiano Pecorelli, Fabio Palomba, Foutse Khomh, Andrea De Lucia: Developer-Driven Code Smell Prioritization. MSR 2020: 220-231      
[2] https://github.com/mauricioaniche/ckhttps://github.com/mauricioaniche/ck/       
[3] Davide Spadini, Maurício Finavaro Aniche, Alberto Bacchelli: PyDriller: Python framework for mining software repositories. ESEC/SIGSOFT FSE 2018: 908-911       
