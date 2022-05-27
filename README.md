# RAW_denoise

## 2022中兴捧月图像去噪方案    
### 思路    
### 参数分析    
为更好地完场图像去噪任务，根据Paper with code上的性能列表，对近两年的Image Denoising on SIDD(Smartphone Image Denoising Dataset)相关工作进行研究与分析。         
参数与性能统计分析：    
![image](https://github.com/zdyshine/RAW_denoise/blob/main/canshu.png)
可以看出，尽管NAFNet在psnr和ssim上均有着最优性能，但其过大的参数量令人望而却步。在实际应用中，仅依靠堆砌网络复杂度来得到微弱的性能提升也是不可取的。综合考虑，Restormer方法在保持极少参数的同时也有着良好的性能，以此方法为基础设计网络或许是一种正确的选择。    
### 网络设计    
级联Restormer网络结构    
![image](https://github.com/zdyshine/RAW_denoise/blob/main/net.png)

#### code说明    
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
 
