# ID2223-FID3020---Lab-2
ID2223/FID3020 - Lab 2

Note that we used this notebook as a base for our finetuning notebook: 
https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=IqM-T1RTzY6C


# Hugginface Space

All the models have been converted to gguf format and the brainstorming idea tool that we have developed work on a free CPU but due to it being a bit not user friendly we offer two ways to interact with our own creative brainstorming tool. One on a free hugginface provided CPU and one on a much faster GPU.

CPU: 

GPU: 

## Task 2
Model: Llama-3.2-1B-Instruct

#Data split
For training and evaluating we used the first 15k examples in the FineTome-100k dataset. The dataset was split into: 
- 12k training dataset
- 1.5k validation dataset
- 1.5k test dataset

# Baseline
As a baseline we primarily used the proposed hyperparameters from the provided notebook. This resulted in these validation losses:

| Step| Training Loss| Validation Loss |
|-----|--------------|-----------------|
| 250 |    0.964800  |    0.879919  |
| 500 |    0.895100  |    0.862609  |
| 750 |    0.889500  |    0.850912  |
| 1000|    0.841500  |    0.843649  |
| 1250|    0.839500  |    0.838236  |
| 1500|    0.827600  |    0.835800  |

We used these hyperparameters for the baseline: r = 16, lora_alpha =16 , learning_rate = 2e-4, warmup_steps = 100, num_train_epochs = 1
(Note that more of the parameters can be found in the notebook)

# Model centric approach
For improving our model we used a model centric approach by completing a small grid search over the learning rate (tune hyperparameters). We searched over these learning rates:
- 2e-4 (baseline)
- 1e-4
- 3e-4

We selected the learning rate yielded the lowest Validation loss and then also search over these parameters:
- r= 16, lora_alpha=16 
- r = 32, lora_alpha= 32
  
r changes the number of trainable parameter, and lora_alpha is a scaling factor.

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

We noticed that, by increasing the learning rate, we gained some improvements in the loss of the model.
The baseline parameters (LR= 2e-4) had a final validation loss of 0.8358, while a learning rate of 3e-4 gave 0.8298 in validation loss. 

While keeping the best performance learning rate we changed to: r = 32 and lora_alpha = 32. These results were obtained:

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 250  | 0.963600      | 0.877421        |
| 500  | 0.892600      | 0.859224        |
| 750  | 0.883300      | 0.844791        |
| 1000 | 0.830100      | 0.834480        |
| 1250 | 0.825700      | 0.825617        |
| 1500 | 0.817700      | 0.821687        |

Increasing the r decreased the validation loss, which is expected due to more trainable parameters (although it is not necessarily always better with a higher r).
After completing our small hyper parameter search we resulted in an improvement of ~1.5 percentages on the validation loss.

Even though the validation loss was lowest in the last condition, we decided to keep r = 16 and lora_alpha= 16 for the test set evaluation. This decision was made due to longer training times with a higher r. 

For the final comparison on the holdout testset, we compared:
- The baseline model (with learning_rate = 2e-4, r = 16, lora_alpha= 16)
- The model trained on tuned hyperparameters (with learning_rate = 3e-4, r = 16, lora_alpha= 16)

Baseline: 

| Step | Training Loss | Test Loss |
|------|---------------|-----------------|
| 250  | 0.964800      | 0.863950        |
| 500  | 0.895100      | 0.845884        |
| 750  | 0.889500      | 0.835044        |
| 1000 | 0.841500      | 0.827129        |
| 1250 | 0.839400      | 0.821827        |
| 1500 | 0.827500      | 0.819001        |


Learning rate: 3e-4

| Step | Training Loss | Test Loss |
|------|---------------|-----------------|
| 250  | 0.964300      | 0.862609        |
| 500  | 0.893100      | 0.843696        |
| 750  | 0.886000      | 0.831871        |
| 1000 | 0.837000      | 0.822710        |
| 1250 | 0.833400      | 0.816040        |
| 1500 | 0.824400      | 0.812596        |

This yielded in an improvement of ~0.65% in loss. It is important that the loss was lower on the test set aswell to show that the performance increase not only holds for the validation set (which was used for tuning the parameters). We believe this test loss could be reduced furhter with higher more lora layers (higher r) as increasing r to 32 reduced validation loss. 

Furthermore, tuning other parameters, such as the number of epochs and weight decay, could have resultet in a better final model. This was not, however, explored further. 

Model-centric improvements that could have been used includes:
- Increasing the number of epochs
- Implementing early stopping (especially important if the number of epochs are high)
- Tuneing more hyperparameters (such as experimenting with weight decay, batch size, LoRA dropout, warmup_steps, etc)
- Experimenting with different schedulars, different LoRA target modules, using rank-stabilized LoRA (setting use_rslora to True) to stabilize training, different optimizers, 
- Testing more foundation models

We also tested Llama-3.2-3B-Instruct on the same test set (see the following section).

# Different foundation model 
We also tried to use another model and finetune on the FineTome-100k dataset with the same split mentioned before.
Model: Llama-3.2-3B-Instruct

| Step | Training Loss | Test Loss |
|------|---------------|-----------------|
| 250  | 0.836200      | 0.741301        |
| 500  | 0.750000      | 0.724180        |
| 750  | 0.759500      | 0.713102        |
| 1000 | 0.723400      | 0.704507        |
| 1250 | 0.707000      | 0.698721        |
| 1500 | 0.704500      | 0.695843        |

We can see that the results from the bigger model were better than the smaller 1b model. When testing the model on the UI on huggingface (CPU), we noticed an improvement in response quality. However, the response time also increased compared to using the 1B model. As we use the model to help with brainstoorming, a lower response time was deemed more important. For our brainstoorming task, the response quality does not have to be maximized. It is more about giving the user ideas fast. Therefore, we decided that the smaller model was more suitable for our brainstorming task.


# Data centric approach
When using a data centric approach to improve the model, it is important to find or create datasets that maps well to the specific task the final model is ment to perform. While the quality of the dataset is definetly important, it is also important that the dataset is suitable for the specific task.

There are many high quality dataset that can be used for fine-tuning. Based on our idea to create a brainstorming llm to help brainstoorming ideas, we decided to train a model on this brainstorming dataset: 

https://huggingface.co/datasets/Wanfq/Explore_Instruct_Brainstorming_10k

We ended up using a model trained on this dataset for our brainstoorming function, and a model finetuned on the FineTome dataset for refining and evaluating ideas. When fine-tuning on the brainstorming dataset, 8K examples were used for training and we used the same hyperparameter settings as for the final model that was fine-tuned on FineTome-100k. 

Due to this being another dataset it was very difficult to compare the loss between the models, but we performed a small qualitatative comparison and generally found (subjectivly) that the brainstoorming answers from the model trained on the brainstoorming dataset were slightly better and the model was also better at formatting the ideas into bulletpoints. (We also let ChatGPT vote on four examples generated from each model on the same four prompts, and it perfered the brainstoorming answers from the model trained on the brainstoorming dataset 3/4 times (see the "small comparison" file if you are interested).)

If there are too few high quality datasets in the domain one is interested in, a distillation approach could be used. A better, larger model can used to generate either answers to an unlabelled dataset or quesion-answer pairs in the relevant domain. A high quality LLM like GPT5 could be used to create a dataset in the relevant domain.  

Other data-centric ways of improving the model could include:
- Filtering the dataset to exclude low-quality training examples
- Removing duplicates
- Filtering the dataset to only include relevant examples


# UI
Our UI is made for brainstoorming and lets the user create different ideas and write about them. Then they can choose to refine the idea, brainstoorm and evaluate the ideas using an LLM. 
The free huggingface CPU is currently used, which makes the model quite slow. We tried using the full models in the beginning, but found that the inference time was too high. Therefore, the models were exported to GGUF & quantized using q4_k_m. This improved the speed and enabled us to run the models on free CPU with limited RAM. It should be noted that this could impact the quality negativly, but we did not observe a noticeable difference. Furthermore, we decided to use finetuned Llama-3.2-1B-Instruct models instead of finetuned Llama-3.2-3B-Instruct models as this improved speed, which is very important for a brainstoorming assistant. Úsing a GPU for inference would have resulted in much faster responses and would have allowed for using a larger model. 
