# RAW_denoise

## 2022中兴捧月图像去噪方案    
### 思路    
为更好地完场图像去噪任务，根据Paper with code上的性能列表（图2.1），我对近两年的Image Denoising on SIDD(Smartphone Image Denoising Dataset)相关工作进行研究与分析。     
参数与性能统计分析：![image](https://github.com/FishInWater-1999/GithubUseTest/blob/master/bac_3.jpg)
step1:    
执行extract_subimages.py进行数据切块，加速训练。    
step2:    
执行train.py进行训练。    
step3:    
执行test.py进行测试。    
注意，下载的训练数据解压后，需要把ground truth文件夹改为ground_truth    
文件夹:    
|—dataset    
>|—ground_truth    
>>|—...dng   
    
>|—noisy       
>>|—...dng     
    
    
crop之后:    
|—crop    
>|—ground_truth_crop    
>>|—0    
>>>|—...dng    
    
>>|—1    
>>|—...    
>>|—97    
    
>|—noisy_crop    
>>|—0    
>>>|—...dng    
    
>>|—1    
>>|—...    
>>|—97    
 
