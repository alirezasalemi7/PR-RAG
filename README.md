# Plan-and-Refine: Diverse and Comprehensive Retrieval-Augmented Generation

This repository contains the codes and packages for the paper titled [Plan-and-Refine: Diverse and Comprehensive Retrieval-Augmented Generation](https://arxiv.org/abs/2504.07794)

This paper studies the limitations of (retrieval-augmented) large language models (LLMs) in generating diverse and comprehensive responses, and introduces the Plan-and-Refine (P&R) framework based on a two phase system design. In the global exploration phase, P&R generates a diverse set of plans for the given input, where each plan consists of a list of diverse query aspects with corresponding additional descriptions. This phase is followed by a local exploitation phase that generates a response proposal for the input query conditioned on each plan and iteratively refines the proposal for improving the proposal quality. Finally, a reward model is employed to select the proposal with the highest factuality and coverage. We conduct our experiments based on the ICAT evaluation methodology--a recent approach for answer factuality and comprehensiveness evaluation. Experiments on the two diverse information seeking benchmarks adopted from non-factoid question answering and TREC search result diversification tasks demonstrate that P&R significantly outperforms baselines, achieving up to a 13.1% improvement on the ANTIQUE dataset and a 15.41% improvement on the TREC dataset. Furthermore, a smaller scale user study confirms the substantial efficacy of the P&R framework.


# Preparing data

The first step is to load the antique dataset and prepare it to be useful for our experiments. You can do this by running the following script:

```shell
python utils/load_antique.py \
    --train_addr /*address to where train data will be saved*/ \
    --test_addr /*address to where test data will be saved*/ \
    --corpus_addr /*address to where corpus will be saved*/
```

# Inference with P&R

To run P&R, you need to do the following steps:

## Generating plan for answering the query

In this step, we use the planner model to generate a diverse set of plans. You can use the following script for this:

```shell
python plan_queries.py \
    --model_addr /*address to checkpoint of the planner model, by default "google/gemma-2-2b-it" is suggested*/ \
    --inputs_addr /*address to the test queries*/ \
    --output_addr /*address to where the plans should be saved*/ \
    --temperature /*planning temperature to increase randomness in plan, suggested value is 0.7*/ \
    --max_tokens 4096 \
    --num_generated_outputs /*number of plans to be generated, we set this to 32*/ \
```

## Retrieving information for plans

In this step, we use a retrieval model to retrieve the information required for excecuting each plan. To do this, we use the following script:

```shell
python retrieval/retriever.py \
    --output_file /*address to where the plans with retrieved information will be stored*/ \
    --input_questions /*address to the test queries*/ \
    --retriever "Snowflake/snowflake-arctic-embed-l" \
    --input_plans /*address to the plans from previous step*/ \
    --n_retrieve /*num documents to be retrieved, we use 40*/ \
    --corpus_path /*address to the corpus*/
```

## Generate initial responses for each plan

In this step, we generate an initial response for each plan, using the following script:

```shell
python generate_with_plan.py \
    --model_addr /*address to generator model, by default "google/gemma-2-2b-it" is suggested*/ \
    --inputs_addr /*address to where the plans with retrieved information is stored*/ \
    --output_addr /*address to where the outputs should be stored*/ \
    --temperature 0.0 \
    --max_tokens 4096 \
    --num_generated_outputs 1 \
    --num_contexts 40 \
    --for_train_or_global_search \
```

## Local Exploitation

In this step, each generated response using the initial plans will go through a series of refinment steps to imporve them, using the following script:

```shell
python local_search.py \
    --model_addr /*address to the refiner model trained checkpoint*/ \
    --inputs_addr /*address to where the plans with retrieved information is stored*/ \
    --initial_response_addr  /*address to where the initial responses are stored*/ \
    --output_addr /*address to where the outputs of local search should be stored*/ \
    --temperature 0.0 \
    --max_tokens 4096 \
    --num_iterations /*number of refinment iterations, suggested to be 32*/
```

## Global exploration

This is the final step, where a trained reward model is used to select the response with the highest reward, as the final response. This can be seen as a global search over all the previous generated responses:

```shell
python global_search.py \
    --model_addr /*address to the checkpoint of the reward model*/ \
    --inputs_addr /*address to where the plans with retrieved information is stored*/ \
    --responses_addr /*address to where the outputs of local search is stored*/ \
    --initial_response_addr  /*address to where the initial responses are stored*/ \
    --output_addr /*address to where the final outputs should be stored*/ \
    --max_tokens 8192 
```

# Evaluation using ICAT

To evaluate the responses using ICAT score, we use the following script:

```shell
python icat_score.py \
    --corpus_path /*address to the corpus*/ \
    --queries /*address to the questions*/ \
    --responses /*address to the responses*/ \
    --output /*address to where the scores should be stored*/ \
    --aggregate \
```

# Training models

## Training planner model

The first step in training the planning model is to generate a diverse set of plans, produce a response for each plan, evaluate the responses, and select those that exceed a certain threshold. The model is then trained on these high-quality, self-generated plans. To do this, we first generate a diverse set of plans:

```shell
python plan_queries.py \
    --model_addr "google/gemma-2-2b-it" \
    --inputs_addr /*address to the training questions*/ \
    --output_addr /*address to the location where plans should be stored*/ \
    --temperature 0.7 \
    --max_tokens 4096 \
    --num_generated_outputs 32 \
```

Then, the next step is to retrieve informtaion for each generated plan:

```shell
python retrieval/retriever.py \
    --output_file /*address to where the plans with retrieved information will be stored for training*/ \
    --input_questions /*address to the train queries*/ \
    --retriever "Snowflake/snowflake-arctic-embed-l" \
    --input_plans /*address to the plans from previous step*/ \
    --n_retrieve /*num documents to be retrieved, we use 40*/ \
    --corpus_path /*address to the corpus*/
```


Then, the next step is to generate responses using these plans:

```shell
python generate_with_plan.py \
    --model_addr /*address to generator model, by default "google/gemma-2-2b-it" is suggested*/ \
    --inputs_addr /*address to where the plans with retrieved information is stored*/ \
    --output_addr /*address to where the outputs should be stored for training*/ \
    --temperature 0.0 \
    --max_tokens 4096 \
    --num_generated_outputs 1 \
    --num_contexts 40 \
    --for_train_or_global_search \
```

Then, the next step is to evaluate the generated responses:

```shell
python icat_score.py \
    --corpus_path /*address to the corpus*/ \
    --queries /*address to the train questions*/ \
    --responses /*address to the generated responses*/ \
    --output /*address to where the scores should be stored for training*/ \
    --train \
```

Then, we sample the examples for training the planner:

```shell
python sampling/creating_training_data_planning.py \
    --score_file /*address to the generated score files*/ \
    --input_plans /*address to where the plans with retrieved information is stored*/ \
    --percentile /*the top percent of plans to be kept for training, we suggest 0.95*/ \
    --training_data /*address to where the training data should be stored*/ \
```

Finally, we can train the model:

```shell
python train_gemma_planning.py \
    --data_addr /*address to where the training data is stored*/ \
    --model_addr "google/gemma-2-2b-it" \
    --output_dir /*address to where the checkpoints should be stored*/ \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --weight_decay 0.0001 \
    --max_steps 1000 \
    --save_steps 200 \
    --warmup_steps 50 \
    --max_seq_length 4096 \
```

## Training editor (refiner) model

To train the editor model for local search, we should take the several steps. First, we need to generate a plan for each training question:

```shell
python plan_queries.py \
    --model_addr /*address to the trained planner mpdel*/ \
    --inputs_addr /*address to the training questions*/ \
    --output_addr /*address to the location where plans should be stored*/ \
    --temperature 0.0 \
    --max_tokens 4096 \
    --num_generated_outputs 1 \
```

Then, the next step is to retrieve informtaion for each generated plan:

```shell
python retrieval/retriever.py \
    --output_file /*address to where the plans with retrieved information will be stored for training*/ \
    --input_questions /*address to the train queries*/ \
    --retriever "Snowflake/snowflake-arctic-embed-l" \
    --input_plans /*address to the plans from previous step*/ \
    --n_retrieve /*num documents to be retrieved, we use 40*/ \
    --corpus_path /*address to the corpus*/
```

Then, for a generated plan, we should generate a set of responses that can be generated from each plan:

```shell
python generate_with_plan.py \
    --model_addr /*address to generator model, by default "google/gemma-2-2b-it" is suggested*/ \
    --inputs_addr /*address to where the plans with retrieved information is stored*/ \
    --output_addr /*address to where the outputs should be stored for training*/ \
    --temperature 0.7 \
    --max_tokens 4096 \
    --num_generated_outputs /*number of generated responses, we suggest 32*/ \
    --num_contexts 40 \
```

Then, the next step is to evaluate the generated responses:

```shell
python icat_score.py \
    --corpus_path /*address to the corpus*/ \
    --queries /*address to the train questions*/ \
    --responses /*address to the generated responses*/ \
    --output /*address to where the scores should be stored for training*/ \
    --train \
```

Then, we sample the examples for training the editor:

```shell
python sampling/creating_training_data_local_search.py \
    --score_file /*address to the generated score files*/ \
    --input_plans /*address to where the plans with retrieved information is stored*/ \
    --response_file /*address to the response files*/ \
    --training_data /*address to where the training data should be stored*/ \
    --threshold /*the threshold for difference between positive and negative examples, suggested to be 0.1*/ \
    --num_samples /*the number samples to keep maximum, suggested to be 8*/ \
```

Finally, we can train the editor model:

```shell
python train_gemma_editor.py \
    --data_addr /*address to where the training data is stored*/ \
    --model_addr "google/gemma-2-2b-it" \
    --output_dir /*address to where the checkpoints should be stored*/ \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --weight_decay 0.0001 \
    --max_steps 1000 \
    --save_steps 200 \
    --warmup_steps 50 \
    --max_seq_length 4096 \
```

## Training the reward model for global search

In order to train the reward model, we can use the already generated data for training planner. Then, the next step is to do sampling for training the reward model:

```shell
python sampling/creating_training_data_global_search.py \
    --score_file /*address to the generated score files*/ \
    --input_plans /*address to where the plans with retrieved information is stored*/ \
    --response_file /*address to the response files*/ \
    --training_data /*address to where the training data should be stored*/ \
    --threshold /*the threshold for difference between positive and negative examples, suggested to be 0.1*/ \
    --num_samples /*the number samples to keep maximum, suggested to be 8*/ \
```

Finally, we can train the reward model:

```shell
python train_reward_model_bert.py \
    --train_data_address /*address to where the training data is stored*/ \
    --model_address "answerdotai/ModernBERT-base" \
    --output_dir /*address to where the checkpoints should be stored*/ \
    --max_len_input 4096 \
    --do_train \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --weight_decay 0.0 \
    --num_train_epochs 10 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --save_strategy epoch \
```

# Reference

```
@misc{salemi2025planandrefinediversecomprehensiveretrievalaugmented,
      title={Plan-and-Refine: Diverse and Comprehensive Retrieval-Augmented Generation}, 
      author={Alireza Salemi and Chris Samarinas and Hamed Zamani},
      year={2025},
      eprint={2504.07794},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.07794}, 
}
```