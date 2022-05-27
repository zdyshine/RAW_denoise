# RAW_denoise

#### 2022中兴捧月图像去噪方案    
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
>>|—*.dng    
		
        |—noisy       
                |—*.dng     
    
crop之后:    
|—crop    
	|—ground_truth_crop    
		|—0    
		        |—*.dng
		|—1    
		|—...    
		|—97    
	|—noisy_crop    
		|—0    
		        |—*.dng    
		|—1    
		|—...    
		|—97    
|—crop_val    
	|—ground_truth_crop    
		|—0    
		|—1    
	|—noisy_crop    
		|—0    
		|—1    
