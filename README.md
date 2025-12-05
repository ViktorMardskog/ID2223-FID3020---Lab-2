# ID2223-FID3020---Lab-2
ID2223/FID3020 - Lab 2


##Task 2

#Data split
For training and evaluating we used the first 15k examples in the FineTome-100k dataset. The dataset was split into: 
- 12k training dataset
- 1.5k validation dataset
- 1.5k test dataset

#Baseline
As a baseline we used the proposed hyperparameters from the provided notebook. This resulted in 

| Step| Training Loss| Validation Loss |
|-----|--------------|-----------------|
| 250 |    -  |    -  |
| 500 |    -  |    -  |
| 750 |    -  |    -  |
| 1000|    -  |    -  |
| 1250|    -  |    -  |
| 1500|    -  |    -  |

| Step| Training Loss| Validation Loss |
|-----|--------------|-----------------|
| 250 |    0.964800  |    0.879919  |
| 500 |    0.895100  |    0.862609  |
| 750 |    0.889500  |    0.850912  |
| 1000|    0.841500  |    0.843649  |
| 1250|    0.839500  |    0.838236  |
| 1500|    0.827600  |    0.835800  |

#Model centric approach
For improving our model we used a model centric approach completing a grid search over the learning rate and also tested different amounts of trainable parameters (r) and the scaling factor (lora_alpha)

While keeping r = 16 and lora_alpha = 16 the learning rate search yeilded:
Learning rate = 1e-4

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 250  | 0.970600      | 0.886649        |
| 500  | 0.901700      | 0.870335        |
| 750  | 0.898500      | 0.860288        |
| 1000 | 0.852600      | 0.854131        |
| 1250 | 0.851800      | 0.850037        |
| 1500 | 0.836200      | 0.848186        |

Learning rate = 3e-4

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 250  | 0.964300      | 0.878334        |
| 500  | 0.893100      | 0.860638        |
| 750  | 0.886100      | 0.847848        |
| 1000 | 0.837100      | 0.839488        |
| 1250 | 0.833500      | 0.832875        |
| 1500 | 0.824500      | 0.829798        |

We noticed that by increasing the learning rate we gained some improvements in the loss of the model.
Decreasing from 0.8358 to 0.8298 

While keeping the best performance learning rate we changed r = 32 and lora_alpha = 32 

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 250  | 0.963600      | 0.877421        |
| 500  | 0.892600      | 0.859224        |
| 750  | 0.883300      | 0.844791        |
| 1000 | 0.830100      | 0.834480        |
| 1250 | 0.825700      | 0.825617        |
| 1500 | 0.817700      | 0.821687        |



