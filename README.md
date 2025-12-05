# ID2223-FID3020---Lab-2
ID2223/FID3020 - Lab 2


## Task 2
Model: Llama-3.2-1B-Instruct
#Data split
For training and evaluating we used the first 15k examples in the FineTome-100k dataset. The dataset was split into: 
- 12k training dataset
- 1.5k validation dataset
- 1.5k test dataset

# Baseline
As a baseline we used the proposed hyperparameters from the provided notebook. This resulted in 

| Step| Training Loss| Validation Loss |
|-----|--------------|-----------------|
| 250 |    0.964800  |    0.879919  |
| 500 |    0.895100  |    0.862609  |
| 750 |    0.889500  |    0.850912  |
| 1000|    0.841500  |    0.843649  |
| 1250|    0.839500  |    0.838236  |
| 1500|    0.827600  |    0.835800  |

# Model centric approach#
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

Increasing the amount of lora layers decreased the loss which is expected due to more trainable parameters.
After completing our small hyper parameter search we resulted in an improvement of ~1.5 percentages on the validation loss.

Due to longer training times with a higher r we decided to evaluate only with the new learning rate: 

Baseline: 

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 250  | 0.964800      | 0.863950        |
| 500  | 0.895100      | 0.845884        |
| 750  | 0.889500      | 0.835044        |
| 1000 | 0.841500      | 0.827129        |
| 1250 | 0.839400      | 0.821827        |
| 1500 | 0.827500      | 0.819001        |


Learning rate: 3e-4

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 250  | 0.964300      | 0.862609        |
| 500  | 0.893100      | 0.843696        |
| 750  | 0.886000      | 0.831871        |
| 1000 | 0.837000      | 0.822710        |
| 1250 | 0.833400      | 0.816040        |
| 1500 | 0.824400      | 0.812596        |

This yielded in an improvement of ~0.65% in loss. We believe this loss could be reduced furhter with higher more lora layers (higher r) and of course longer training.

# Data centric approach
Based on our idea to create a brainstorming llm to help out generate and evaluate ideas we decided to train a model on brainstorming dataset: 

https://huggingface.co/datasets/Wanfq/Explore_Instruct_Brainstorming_10k

Due to this being another dataset it was very difficult to compare the loss between the models but in our opinion the ideas were better and the model was also better at formatting the ideas into bulletpoints. 

# Different foundation model 
We also tried to use another model and finetune on the FineTome-100k dataset with the same split mentioned before.
Model: Llama-3.2-3B-Instruct

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 250  | 0.836200      | 0.741301        |
| 500  | 0.750000      | 0.724180        |
| 750  | 0.759500      | 0.713102        |
| 1000 | 0.723400      | 0.704507        |
| 1250 | 0.707000      | 0.698721        |
| 1500 | 0.704500      | 0.695843        |

We can see that the results from the bigger model was way better then the smaller 1b model but when testing this model at inference time on huggingface we decided that the smaller model was more suitable for our brainstorming task.



